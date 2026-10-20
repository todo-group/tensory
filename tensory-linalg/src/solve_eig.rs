use alloc::vec;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeErr, RuntimeFor},
    },
    mapper::{
        AxisMapper, EquivGroupMapper, EquivGroupedAxes, SolveConf, SolveGroupedMapper, SplittyErr,
    },
    repr::TensorTupleRepr,
    tensor::{BoundTensorTuple, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a solve-eigenvalue operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SolveEigRepr<A: TensorTupleRepr<1>> {
    a: A,
    axes_split: EquivGroupedAxes<2>,
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<2> for SolveEigRepr<A> {
    fn naxes_array(&self) -> [usize; 2] {
        let (_, [left_axes, _right_axes]) = self.axes_split.clone().into_raw();
        [left_axes.len() + 1, 2]
    }
}

impl<A: TensorTupleRepr<1>> tensory_core::concept::task::IsTask for SolveEigRepr<A> {}

impl<A: TensorTupleRepr<1>> SolveEigRepr<A> {
    /// Creates a solve-eigenvalue representation without validating metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`, and the enclosing tuple's
    /// mappers must match the two output representations.
    pub unsafe fn from_raw_unchecked(a: A, axes_split: EquivGroupedAxes<2>) -> Self {
        Self { a, axes_split }
    }

    /// Decomposes the representation into its input and axis metadata.
    pub fn into_raw(self) -> (A, EquivGroupedAxes<2>) {
        (self.a, self.axes_split)
    }
}

/// Lazy solve-eigenvalue construction for ordinary tensors.
pub trait TensorSolveEigExt: ToTensor {
    /// Builds a solve-eigenvalue task with separate IDs for the bonds.
    fn solve_eig_with_more_ids<Q>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, SolveEigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>;

    /// Builds a solve-eigenvalue task using shared bond IDs.
    fn solve_eig<Q>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, SolveEigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone;
}

impl<T: ToTensor> TensorSolveEigExt for T {
    fn solve_eig_with_more_ids<Q>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, SolveEigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
    {
        let (raw, [legs]) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.equiv_split(query).map_err(SplittyErr::Split)?;
        let [v_legs, d_legs] = unsafe {
            grouped.solve(SolveConf::from_raw_unchecked(
                [[true, false], [false, false]],
                vec![(0, v_vd_leg), (1, d_vd_leg), (1, d_dvinv_leg)],
            ))
        }
        .map_err(SplittyErr::Use)?;
        let repr = unsafe { SolveEigRepr::from_raw_unchecked(raw, axes_split) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, [v_legs, d_legs]) })
    }

    fn solve_eig<Q>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, SolveEigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
    {
        self.solve_eig_with_more_ids(query, vd_leg.clone(), vd_leg.clone(), dvinv_leg)
    }
}

/// Runtime-bound solve-eigenvalue construction and execution.
pub trait BoundTensorSolveEigExt: ToBoundTensorTuple<1> {
    /// Builds and executes a solve-eigenvalue task with separate bond IDs.
    fn solve_eig_with_more_ids<Q, E>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
                2,
                SolveEigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        Self::Runtime: RuntimeFor<SolveEigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
            2,
            SolveEigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;

    /// Builds and executes a solve-eigenvalue task using shared bond IDs.
    fn solve_eig<Q, E>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
                2,
                SolveEigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<SolveEigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
            2,
            SolveEigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorSolveEigExt for T {
    fn solve_eig_with_more_ids<Q, E>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
                2,
                SolveEigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        Self::Runtime: RuntimeFor<SolveEigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
            2,
            SolveEigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (a, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = a
            .solve_eig_with_more_ids(query, v_vd_leg, d_vd_leg, d_dvinv_leg)
            .map_err(RuntimeErr::Defer)?;
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensorTuple::from_raw(result, runtime))
    }

    fn solve_eig<Q, E>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvinv_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
                2,
                SolveEigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<SolveEigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<SolveEigRepr<Self::Repr>>>::Mk,
            2,
            SolveEigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        self.solve_eig_with_more_ids(query, vd_leg.clone(), vd_leg.clone(), dvinv_leg)
    }
}
