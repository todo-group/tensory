use alloc::vec;

use tensory_core::{
    bound_tensor::{BoundTensor, Runtime, RuntimeError, ToBoundTensor},
    mapper::{AxisMapper, DecompConf, DecompGroupedMapper, GroupMapper, GroupedAxes, SplittyError},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of SVD operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait SvdCtxImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding U.
    type U: TensorRepr; // axis order: a, <from A for U>
    /// The type of the result tensor representation corresponding S.
    type S: TensorRepr; // axis order: a, b
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr; // axis order: b, <from A for V>
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
pub trait SvdCtx<A: TensorRepr>: SvdCtxImpl<A> {
    fn svd(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err>;
}
impl<C: SvdCtxImpl<A>, A: TensorRepr> SvdCtx<A> for C {
    fn svd(
        self,
        a: A,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.svd_unchecked(a, axes_split) }
    }
}

pub struct TensorSvd<A: TensorRepr, B: AxisMapper> {
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

impl<A: TensorRepr, B: AxisMapper, C: SvdCtx<A>> TensorTask<C> for TensorSvd<A, B> {
    type Output = Result<(Tensor<C::U, B>, Tensor<C::S, B>, Tensor<C::V, B>), C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let (u, s, v) = unsafe { ctx.svd_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(u, self.u_legs) },
            unsafe { Tensor::from_raw_unchecked(s, self.s_legs) },
            unsafe { Tensor::from_raw_unchecked(v, self.v_legs) },
        ))
    }
}

pub trait TensorSvdExt<A: TensorRepr, M: AxisMapper>: Sized {
    fn svd_with_more_ids<Q>(
        self,
        query: Q,
        u_us_leg: M::Id,
        s_us_leg: M::Id,
        s_sv_leg: M::Id,
        v_sv_leg: M::Id,
    ) -> Result<TensorSvd<A, M>, SplittyError<M::Err, <M::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        M: GroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 3>;
    fn svd<Q>(
        self,
        set: Q,
        us_leg: M::Id,
        sv_leg: M::Id,
    ) -> Result<TensorSvd<A, M>, SplittyError<M::Err, <M::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        M: GroupMapper<2, Q>,
        M::Grouped: DecompGroupedMapper<2, 3>,
        M::Id: Clone;
}

impl<T: ToTensor> TensorSvdExt<T::Repr, T::Mapper> for T {
    fn svd_with_more_ids<Q>(
        self,
        query: Q,
        u_us_leg: <T::Mapper as AxisMapper>::Id,
        s_us_leg: <T::Mapper as AxisMapper>::Id,
        s_sv_leg: <T::Mapper as AxisMapper>::Id,
        v_sv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorSvd<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as GroupMapper<2, Q>>::Err,
            <<T::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: GroupMapper<2, Q>,
        <T::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.split(query).map_err(SplittyError::Split)?;
        let [u_legs, s_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, u_us_leg), (1, s_us_leg)),
                    ((1, s_sv_leg), (2, v_sv_leg)),
                ],
            ))
        }
        .map_err(SplittyError::Use)?;
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
        query: Q,
        us_leg: <T::Mapper as AxisMapper>::Id,
        sv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorSvd<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as GroupMapper<2, Q>>::Err,
            <<T::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: GroupMapper<2, Q>,
        <T::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <T::Mapper as AxisMapper>::Id: Clone,
    {
        self.svd_with_more_ids(query, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    }
}

pub trait SvdWithOptionRuntime<A: TensorRepr, O>: Runtime {
    type Ctx: SvdCtxImpl<A>;
    fn svd_ctx(&self, opt: O) -> Self::Ctx;
}

pub trait BoundTensorSvdExt: ToBoundTensor {
    fn svd_with_more_ids<Q, O>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        (
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::U,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::S,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::V,
                Self::Mapper,
                Self::Runtime,
            >,
        ),
        RuntimeError<
            SplittyError<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                Self::Repr,
            >>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: SvdWithOptionRuntime<Self::Repr, O>;

    fn svd<Q, O>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        (
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::U,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::S,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::V,
                Self::Mapper,
                Self::Runtime,
            >,
        ),
        RuntimeError<
            SplittyError<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                Self::Repr,
            >>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: SvdWithOptionRuntime<Self::Repr, O>;
}

impl<T: ToBoundTensor> BoundTensorSvdExt for T {
    fn svd_with_more_ids<Q, O>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        (
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::U,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::S,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::V,
                Self::Mapper,
                Self::Runtime,
            >,
        ),
        RuntimeError<
            SplittyError<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                Self::Repr,
            >>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: SvdWithOptionRuntime<Self::Repr, O>,
    {
        let (a, rt) = self.to_bound_tensor().into_raw();

        a.svd_with_more_ids(query, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg)
            .map_err(RuntimeError::Axis)?
            .with(rt.svd_ctx(option))
            .map_err(RuntimeError::Ctx)
            .map(|(u, s, v)| (u.bind(rt.clone()), s.bind(rt.clone()), v.bind(rt)))
    }

    fn svd<Q, O>(
        self,
        set: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        (
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::U,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::S,
                Self::Mapper,
                Self::Runtime,
            >,
            BoundTensor<
                <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                    Self::Repr,
                >>::V,
                Self::Mapper,
                Self::Runtime,
            >,
        ),
        RuntimeError<
            SplittyError<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            <<Self::Runtime as SvdWithOptionRuntime<Self::Repr, O>>::Ctx as SvdCtxImpl<
                Self::Repr,
            >>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: SvdWithOptionRuntime<Self::Repr, O>,
    {
        self.svd_with_more_ids(set, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg, option)
    }
}
