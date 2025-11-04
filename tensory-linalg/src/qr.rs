use alloc::vec;

use tensory_core::{
    mapper::{AxisMapper, DecompConf, DecompError, DecompGroupedMapper, GroupMapper, GroupedAxes},
    repr::{AsViewRepr, TensorRepr},
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of QR operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `QrAxisProvenance` is valid for the given tensor.
pub unsafe trait QrContextImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding Q.
    type Q: TensorRepr; // axis order: a, <from A for Q>
    /// The type of the result tensor representation corresponding R.
    type R: TensorRepr; // axis order: a, b
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs QR operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `QrAxisProvenance` is valid for the given tensor.
    unsafe fn qr_unchecked(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::Q, Self::R), Self::Err>;
}

/// Safe version if QrContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait QrContext<A: TensorRepr>: QrContextImpl<A> {
    fn qr(self, a: A, axes_split: GroupedAxes<2>) -> Result<(Self::Q, Self::R), Self::Err>;
}
impl<C: QrContextImpl<A>, A: TensorRepr> QrContext<A> for C {
    fn qr(self, a: A, axes_split: GroupedAxes<2>) -> Result<(Self::Q, Self::R), Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.qr_unchecked(a, axes_split) }
    }
}

pub struct TensorQr<A: TensorRepr, M: AxisMapper> {
    a: A,
    q_legs: M,
    r_legs: M,
    axes_split: GroupedAxes<2>,
}

impl<A: TensorRepr, M: AxisMapper, C: QrContext<A>> TensorTask<C> for TensorQr<A, M> {
    type Output = Result<(Tensor<C::Q, M>, Tensor<C::R, M>), C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let (q, r) = unsafe { ctx.qr_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(q, self.q_legs) },
            unsafe { Tensor::from_raw_unchecked(r, self.r_legs) },
        ))
    }
}

pub trait TensorQrExt<A: TensorRepr, M: AxisMapper>: Sized {
    fn qr_with_more_ids<Q>(
        self,
        set: Q,
        q_qr_leg: M::Id,
        r_qr_leg: M::Id,
    ) -> Result<TensorQr<A, M>, DecompError<M::Err, <M::Grouped as DecompGroupedMapper<2, 2>>::Err>>
    where
        M: GroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 2>;
    fn qr<Q>(
        self,
        set: Q,
        qr_leg: M::Id,
    ) -> Result<TensorQr<A, M>, DecompError<M::Err, <M::Grouped as DecompGroupedMapper<2, 2>>::Err>>
    where
        M: GroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 2>,
        M::Id: Clone;
}

impl<T: ToTensor> TensorQrExt<T::Repr, T::Mapper> for T {
    fn qr_with_more_ids<Q>(
        self,
        queue: Q,
        q_qr_leg: <T::Mapper as AxisMapper>::Id,
        r_qr_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorQr<T::Repr, T::Mapper>,
        DecompError<
            <T::Mapper as GroupMapper<2, Q>>::Err,
            <<T::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        T::Mapper: GroupMapper<2, Q>,
        <T::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.split(queue).map_err(|e| DecompError::Split(e))?;
        let [q_legs, r_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 1],
                vec![((0, q_qr_leg), (1, r_qr_leg))],
            ))
        }
        .map_err(|e| DecompError::Decomp(e))?;
        Ok(TensorQr {
            a: raw,
            q_legs,
            r_legs,
            axes_split,
        })
    }
    fn qr<Q>(
        self,
        set: Q,
        qr_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorQr<T::Repr, T::Mapper>,
        DecompError<
            <T::Mapper as GroupMapper<2, Q>>::Err,
            <<T::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        <T::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        T::Mapper: GroupMapper<2, Q>,
        <T::Mapper as AxisMapper>::Id: Clone,
    {
        self.qr_with_more_ids(set, qr_leg.clone(), qr_leg)
    }
}
