use alloc::vec;

use tensory_core::tensor::{
    AsViewRepr, DecompConf, DecompError, DecompGroupedBroker, GroupBroker, GroupedAxes, Tensor,
    TensorBroker, TensorRepr,
};

/// Raw context of SVD operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait SvdContextImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding U.
    type U: TensorRepr; // a, <from A for U>
    /// The type of the result tensor representation corresponding S.
    type S: TensorRepr; // a,b
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr; // b, <from A for V>
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs SVD operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn svd_unchecked(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err>;
}

/// Safe version if SvdContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait SvdContext<A: TensorRepr>: SvdContextImpl<A> {
    fn svd(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err>;
}
impl<C: SvdContextImpl<A>, A: TensorRepr> SvdContext<A> for C {
    fn svd(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err> {
        if a.dim() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.svd_unchecked(a, axes_split) }
    }
}

pub struct TensorSvd<A: TensorRepr, B: TensorBroker> {
    a: A,
    u_legs: B,
    s_legs: B,
    v_legs: B,
    axes_split: GroupedAxes<2>,
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

impl<A: TensorRepr, B: TensorBroker> TensorSvd<A, B> {
    // pub fn new<Q>(
    //     a: Tensor<A, B>,
    //     queue: Q,
    //     s_us_leg: B::Id,
    //     s_sv_leg: B::Id,
    //     v_sv_leg: B::Id,
    // ) -> Self {
    //     let (raw, legs) = a.into_raw();

    //     let (intermediate, u_axes) =
    //         LA::extract(legs, u_legs, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg);

    //     Self {
    //         a: raw,
    //         intermediate,
    //         u_axes: u_axes,
    //     }
    // }
    pub fn with<C: SvdContext<A>>(
        self,
        context: C,
    ) -> Result<(Tensor<C::U, B>, Tensor<C::S, B>, Tensor<C::V, B>), C::Err> {
        let a = self.a;
        let axes_split = self.axes_split;

        let (u, s, v) = unsafe { context.svd_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(u, self.u_legs) },
            unsafe { Tensor::from_raw_unchecked(s, self.s_legs) },
            unsafe { Tensor::from_raw_unchecked(v, self.v_legs) },
        ))
    }
}

pub trait TensorSvdExt<A: TensorRepr, B: TensorBroker>: Sized {
    fn svd_with_more_ids<Q>(
        self,
        set: Q,
        u_us_leg: B::Id,
        s_us_leg: B::Id,
        s_sv_leg: B::Id,
        v_sv_leg: B::Id,
    ) -> Result<TensorSvd<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>>
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>;
    fn svd<Q>(
        self,
        set: Q,
        us_leg: B::Id,
        sv_leg: B::Id,
    ) -> Result<TensorSvd<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>>
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>,
        B::Id: Clone;
}

impl<A: TensorRepr, B: TensorBroker> TensorSvdExt<A, B> for Tensor<A, B> {
    fn svd_with_more_ids<Q>(
        self,
        queue: Q,
        u_us_leg: B::Id,
        s_us_leg: B::Id,
        s_sv_leg: B::Id,
        v_sv_leg: B::Id,
    ) -> Result<TensorSvd<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>>
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>,
    {
        let (raw, legs) = self.into_raw();
        let (grouped, axes_split) = legs.split(queue).map_err(|e| DecompError::Split(e))?;
        let [u_legs, s_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, u_us_leg), (1, s_us_leg)),
                    ((1, s_sv_leg), (2, v_sv_leg)),
                ],
            ))
        }
        .map_err(|e| DecompError::Decomp(e))?;
        Ok(TensorSvd {
            a: raw,
            u_legs,
            s_legs,
            v_legs,
            axes_split,
        })
    }
    fn svd<Q>(
        self,
        set: Q,
        us_leg: B::Id,
        sv_leg: B::Id,
    ) -> Result<TensorSvd<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>>
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>,
        B::Id: Clone,
    {
        self.svd_with_more_ids(set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    }
    // pub fn svd<Set>(self, set: Set, us_leg: B::Id, sv_leg: B::Id) -> TensorSvd<Set, B>
    // where
    //     LA: SvdLegAlloc<Set>,
    //     LA::Id: Clone,
    // {
    //     TensorSvd::new(self, set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    // }
}

impl<'a, A: AsViewRepr<'a>, B: TensorBroker + Clone> TensorSvdExt<A::View, B> for &'a Tensor<A, B> {
    fn svd_with_more_ids<Q>(
        self,
        queue: Q,
        u_us_leg: B::Id,
        s_us_leg: B::Id,
        s_sv_leg: B::Id,
        v_sv_leg: B::Id,
    ) -> Result<
        TensorSvd<A::View, B>,
        DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>,
    >
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>,
    {
        let (raw, legs) = self.view().into_raw();
        let (grouped, axes_split) = legs.split(queue).map_err(|e| DecompError::Split(e))?;
        let [u_legs, s_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, u_us_leg), (1, s_us_leg)),
                    ((1, s_sv_leg), (2, v_sv_leg)),
                ],
            ))
        }
        .map_err(|e| DecompError::Decomp(e))?;
        Ok(TensorSvd {
            a: raw,
            u_legs,
            s_legs,
            v_legs,
            axes_split,
        })
    }
    fn svd<Q>(
        self,
        set: Q,
        us_leg: B::Id,
        sv_leg: B::Id,
    ) -> Result<
        TensorSvd<A::View, B>,
        DecompError<B::Err, <B::Grouped as DecompGroupedBroker<2, 3>>::Err>,
    >
    where
        B: GroupBroker<2, Q>,
        B::Grouped: DecompGroupedBroker<2, 3>,
        B::Id: Clone,
    {
        self.svd_with_more_ids(set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    }
    // pub fn svd<Set>(self, set: Set, us_leg: B::Id, sv_leg: B::Id) -> TensorSvd<Set, B>
    // where
    //     LA: SvdLegAlloc<Set>,
    //     LA::Id: Clone,
    // {
    //     TensorSvd::new(self, set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    // }
}
