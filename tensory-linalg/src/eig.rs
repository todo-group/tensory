use alloc::vec;

use tensory_core::{
    mapper::{
        AxisMapper, DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes,
        SplittyError,
    },
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of diagonalization operation: A= V*D*VC.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait EigCtxImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr; // axis order: a, <from A for V>
    /// The type of the result tensor representation corresponding D.
    type D: TensorRepr; // axis order: a, b
    /// The type of the result tensor representation corresponding VC.
    type VC: TensorRepr; // axis order: b, <from A for VC>
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs SVD operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn eig_unchecked(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::V, Self::D, Self::VC), Self::Err>;
}

/// Safe version if SvdContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait EigCtx<A: TensorRepr>: EigCtxImpl<A> {
    fn eig(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::V, Self::D, Self::VC), Self::Err>;
}
impl<C: EigCtxImpl<A>, A: TensorRepr> EigCtx<A> for C {
    fn eig(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::V, Self::D, Self::VC), Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.eig_unchecked(a, axes_split) }
    }
}

pub struct TensorEig<A: TensorRepr, M: AxisMapper> {
    a: A,
    v_legs: M,
    d_legs: M,
    vc_legs: M,
    axes_split: EquivGroupedAxes<2>,
}

// pub unsafe trait SvdLegAlloc<Set>: LegAlloc {
//     type Intermediate;
//     fn extract(
//         a: Self,
//         u_legs: Set,
//         u_us_leg: Self::Id,
//         s_us_leg: Self::Id,
//         s_sv_leg: Self::Id,
//         v_sv_leg: Self::Id,
//     ) -> (Self::Intermediate, Vec<usize>);
//     unsafe fn merge(
//         intermediate: Self::Intermediate,
//         u_provenance: Vec<SvdIsometryAxisProvenance>,
//         s_provenance: SvdSingularAxisOrder,
//         v_provenance: Vec<SvdIsometryAxisProvenance>,
//     ) -> (Self, Self, Self)
//     where
//         Self: Sized;
// }

// impl<A: TensorRepr, M: AxisMapper> TensorEig<A, M> {
//     // pub fn new<Q>(
//     //     a: Tensor<A, B>,
//     //     queue: Q,
//     //     s_us_leg: B::Id,
//     //     s_sv_leg: B::Id,
//     //     v_sv_leg: B::Id,
//     // ) -> Self {
//     //     let (raw, legs) = a.into_raw();

//     //     let (intermediate, u_axes) =
//     //         LA::extract(legs, u_legs, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg);

//     //     Self {
//     //         a: raw,
//     //         intermediate,
//     //         u_axes: u_axes,
//     //     }
//     // }
// }
impl<A: TensorRepr, M: AxisMapper, C: EigCtx<A>> TensorTask<C> for TensorEig<A, M> {
    type Output = Result<(Tensor<C::V, M>, Tensor<C::D, M>, Tensor<C::VC, M>), C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let (v, d, vc) = unsafe { ctx.eig_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(v, self.v_legs) },
            unsafe { Tensor::from_raw_unchecked(d, self.d_legs) },
            unsafe { Tensor::from_raw_unchecked(vc, self.vc_legs) },
        ))
    }
}

pub trait TensorEigExt<A: TensorRepr, M: AxisMapper>: Sized {
    fn eig_with_more_ids<Q>(
        self,
        set: Q,
        v_vd_leg: M::Id,
        d_vd_leg: M::Id,
        d_dvc_leg: M::Id,
        vc_dvc_leg: M::Id,
    ) -> Result<TensorEig<A, M>, SplittyError<M::Err, <M::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        M: EquivGroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 3>;
    fn eig<Q>(
        self,
        set: Q,
        vd_leg: M::Id,
        dvc_leg: M::Id,
    ) -> Result<TensorEig<A, M>, SplittyError<M::Err, <M::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        M: EquivGroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 3>,
        M::Id: Clone;
}

impl<T: ToTensor> TensorEigExt<T::Repr, T::Mapper> for T {
    fn eig_with_more_ids<Q>(
        self,
        queue: Q,
        v_vd_leg: <T::Mapper as AxisMapper>::Id,
        d_vd_leg: <T::Mapper as AxisMapper>::Id,
        d_dvc_leg: <T::Mapper as AxisMapper>::Id,
        vc_dvc_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorEig<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs
            .equiv_split(queue)
            .map_err(|e| SplittyError::Split(e))?;
        let [vc_legs, d_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, v_vd_leg), (1, d_vd_leg)),
                    ((1, d_dvc_leg), (2, vc_dvc_leg)),
                ],
            ))
        }
        .map_err(|e| SplittyError::Use(e))?;
        Ok(TensorEig {
            a: raw,
            v_legs: vc_legs,
            d_legs,
            vc_legs: v_legs,
            axes_split,
        })
    }
    fn eig<Q>(
        self,
        set: Q,
        vd_leg: <T::Mapper as AxisMapper>::Id,
        dvc_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorEig<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <T::Mapper as AxisMapper>::Id: Clone,
    {
        self.eig_with_more_ids(set, vd_leg.clone(), vd_leg, dvc_leg.clone(), dvc_leg)
    }
}
