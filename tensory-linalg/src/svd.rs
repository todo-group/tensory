use alloc::vec::Vec;
use tensory_core::tensor::{DecompAxisOrigin, Tensor, TensorBroker, TensorRepr};

/// Raw context of SVD operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait SvdContextImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding U.
    type U: TensorRepr;
    /// The type of the result tensor representation corresponding S.
    type S: TensorRepr;
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr;
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
        u_axes: &[DecompAxisOrigin],
        s_axes: &[DecompAxisOrigin],
        v_axes: &[DecompAxisOrigin],
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err>;
}

/// Safe version if SvdContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait SvdContext<A: TensorRepr>: SvdContextImpl<A> {
    fn svd(self, a: A, u_axes: &[usize]) -> Result<(Self::U, Self::S, Self::V), Self::Err>;
}
impl<C: SvdContextImpl<A>, A: TensorRepr> SvdContext<A> for C {
    fn svd(
        self,
        a: A,
        u_axes: &[usize],
    ) -> Result<
        (
            Self::U,
            Self::S,
            Self::V,
            Vec<SvdIsometryAxisProvenance>,
            SvdSingularAxisOrder,
            Vec<SvdIsometryAxisProvenance>,
        ),
        Self::Err,
    > {
        // TODO check input and output
        // if input invalid {
        //     panic!();
        // }
        unsafe { self.svd_unchecked(a, u_axes) }
    }
}

pub struct TensorSvd<Set, LA: SvdLegAlloc<Set>, A: TensorRepr> {
    a: A,
    u_axes: Vec<usize>,
    intermediate: LA::Intermediate,
    //legs: Vec<MaybeUninit<LA::Id>>,
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

impl<Set, LA: SvdLegAlloc<Set>, A: TensorRepr> TensorSvd<Set, LA, A> {
    pub fn new(
        a: Tensor<LA, A>,
        u_legs: Set,
        u_us_leg: LA::Id,
        s_us_leg: LA::Id,
        s_sv_leg: LA::Id,
        v_sv_leg: LA::Id,
    ) -> Self {
        let (raw, legs) = a.into_raw();

        let (intermediate, u_axes) =
            LA::extract(legs, u_legs, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg);

        Self {
            a: raw,
            intermediate,
            u_axes: u_axes,
        }
    }
    pub fn with<C: SvdContext<A>>(
        self,
        context: C,
    ) -> Result<(Tensor<LA, C::U>, Tensor<LA, C::S>, Tensor<LA, C::V>), C::Err> {
        let a = self.a;
        let intermediate = self.intermediate;
        let u_axes = self.u_axes;

        let (u, s, v, u_alloc, s_order, v_alloc) = unsafe { context.svd_unchecked(a, &u_axes) }?;

        let (u_legs, s_legs, v_legs) =
            unsafe { LA::merge(intermediate, u_alloc, s_order, v_alloc) };

        Ok((
            unsafe { Tensor::from_raw_unchecked(u, u_legs) },
            unsafe { Tensor::from_raw_unchecked(s, s_legs) },
            unsafe { Tensor::from_raw_unchecked(v, v_legs) },
        ))
    }
}

trait TensorSvdExt<LA: SvdLegAlloc<Set>, Set>
where
    LA::Id: Clone,
{
    fn svd_with_more_ids<Set>(
        self,
        set: Set,
        u_us_leg: LA::Id,
        s_us_leg: LA::Id,
        s_sv_leg: LA::Id,
        v_sv_leg: LA::Id,
    ) -> TensorSvd<Set, LA, T>
    where
        LA: SvdLegAlloc<Set>,
    {
        TensorSvd::new(self, set, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg)
    }
}

impl<M: TensorBroker, T: TensorRepr> Tensor<M, T> {
    pub fn svd_with_more_ids<Set>(
        self,
        set: Set,
        u_us_leg: LA::Id,
        s_us_leg: LA::Id,
        s_sv_leg: LA::Id,
        v_sv_leg: LA::Id,
    ) -> TensorSvd<Set, LA, T>
    where
        LA: SvdLegAlloc<Set>,
    {
        TensorSvd::new(self, set, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg)
    }
    pub fn svd<Set>(self, set: Set, us_leg: LA::Id, sv_leg: LA::Id) -> TensorSvd<Set, LA, T>
    where
        LA: SvdLegAlloc<Set>,
        LA::Id: Clone,
    {
        TensorSvd::new(self, set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    }
}
