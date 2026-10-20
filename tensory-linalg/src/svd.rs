use alloc::vec;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::{AxisMapper, DecompConf, DecompGroupedMapper, GroupMapper, GroupedAxes, SplittyErr},
    repr::TensorTupleRepr,
    tensor::{BoundTensorTuple, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of an SVD operation.
///
/// The representation describes the three output tensors. The input
/// representation and the axis grouping stay in the task, while the output
/// mappers are kept by the enclosing `TensorTuple`.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SvdRepr<A: TensorTupleRepr<1>, O> {
    a: A,
    axes_split: GroupedAxes<2>,
    option: O,
}

unsafe impl<A: TensorTupleRepr<1>, O> TensorTupleRepr<3> for SvdRepr<A, O> {
    fn naxes_array(&self) -> [usize; 3] {
        let (_, [u_axes, v_axes]) = self.axes_split.clone().into_raw();
        [u_axes.len() + 1, 2, v_axes.len() + 1]
    }
}

impl<A: TensorTupleRepr<1>, O> IsTask for SvdRepr<A, O> {}

impl<A: TensorTupleRepr<1>, O> SvdRepr<A, O> {
    /// Creates an SVD representation without validating its metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`, and the enclosing tuple's
    /// three mappers must match the three output representations.
    pub unsafe fn from_raw_unchecked(a: A, axes_split: GroupedAxes<2>, option: O) -> Self {
        Self {
            a,
            axes_split,
            option,
        }
    }

    /// Decomposes the representation into its input, axis metadata, and option.
    pub fn into_raw(self) -> (A, GroupedAxes<2>, O) {
        (self.a, self.axes_split, self.option)
    }
}

/// Lazy SVD construction for ordinary tensors.
pub trait TensorSvdExt: ToTensor {
    /// Builds an SVD task with separate IDs for the two bonds.
    fn svd_with_more_ids<Q>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, ()>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>;

    /// Builds an SVD task using one shared ID for both bonds.
    fn svd<Q>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, ()>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone;

    /// Builds an SVD task using one shared ID for both bonds and a backend option.
    fn svd_with_option<Q, O>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, O>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self: Sized,
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone;

    /// Builds an SVD task with separate IDs and a backend option.
    fn svd_with_more_ids_and_option<Q, O>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, O>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self: Sized,
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>;
}

impl<T: ToTensor> TensorSvdExt for T {
    fn svd_with_more_ids<Q>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, ()>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        self.svd_with_more_ids_and_option(query, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg, ())
    }

    fn svd_with_more_ids_and_option<Q, O>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, O>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        let (raw, [legs]) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.split(query).map_err(SplittyErr::Split)?;
        let [u_legs, s_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, u_us_leg), (1, s_us_leg)),
                    ((1, s_sv_leg), (2, v_sv_leg)),
                ],
            ))
        }
        .map_err(SplittyErr::Use)?;
        let repr = unsafe { SvdRepr::from_raw_unchecked(raw, axes_split, option) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, [u_legs, s_legs, v_legs]) })
    }

    fn svd<Q>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, ()>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
    {
        self.svd_with_more_ids(query, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg)
    }

    fn svd_with_option<Q, O>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        TensorTuple<3, SvdRepr<Self::Repr, O>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
    {
        self.svd_with_more_ids_and_option(query, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg, option)
    }
}

/// Runtime-bound SVD construction and execution.
pub trait BoundTensorSvdExt: ToBoundTensorTuple<1> {
    /// Builds and executes an SVD with separate IDs for the two bonds.
    fn svd_with_more_ids<Q, O, E>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
                3,
                SvdRepr<Self::Repr, O>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: RuntimeFor<SvdRepr<Self::Repr, O>>,
        <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
            3,
            SvdRepr<Self::Repr, O>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;

    /// Builds and executes an SVD using one shared ID for both bonds.
    fn svd<Q, O, E>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
                3,
                SvdRepr<Self::Repr, O>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<SvdRepr<Self::Repr, O>>,
        <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
            3,
            SvdRepr<Self::Repr, O>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorSvdExt for T {
    fn svd_with_more_ids<Q, O, E>(
        self,
        query: Q,
        u_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_us_leg: <Self::Mapper as AxisMapper>::Id,
        s_sv_leg: <Self::Mapper as AxisMapper>::Id,
        v_sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
                3,
                SvdRepr<Self::Repr, O>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: RuntimeFor<SvdRepr<Self::Repr, O>>,
        <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
            3,
            SvdRepr<Self::Repr, O>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (a, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = a
            .svd_with_more_ids(query, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg)
            .map_err(RuntimeErr::Defer)?;
        let (repr, mappers) = task.into_raw();
        let (raw, axes_split, _) = repr.into_raw();
        let repr = unsafe { SvdRepr::from_raw_unchecked(raw, axes_split, option) };
        let task = unsafe { TensorTuple::from_raw_unchecked(repr, mappers) };
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensorTuple::from_raw(result, runtime))
    }

    fn svd<Q, O, E>(
        self,
        query: Q,
        us_leg: <Self::Mapper as AxisMapper>::Id,
        sv_leg: <Self::Mapper as AxisMapper>::Id,
        option: O,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
                3,
                SvdRepr<Self::Repr, O>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<SvdRepr<Self::Repr, O>>,
        <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SvdRepr<Self::Repr, O>>>::Mk,
            3,
            SvdRepr<Self::Repr, O>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        self.svd_with_more_ids(query, us_leg.clone(), us_leg, sv_leg.clone(), sv_leg, option)
    }
}
