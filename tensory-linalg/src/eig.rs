use alloc::vec;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeErr, RuntimeFor},
    },
    mapper::{
        AxisMapper, DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes,
        SplittyErr,
    },
    repr::TensorTupleRepr,
    tensor::{BoundTensorTuple, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a diagonalization operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EigRepr<A: TensorTupleRepr<1>> {
    a: A,
    axes_split: EquivGroupedAxes<2>,
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<3> for EigRepr<A> {
    fn naxes_array(&self) -> [usize; 3] {
        let (_, [left_axes, right_axes]) = self.axes_split.clone().into_raw();
        [left_axes.len() + 1, 2, right_axes.len() + 1]
    }
}

impl<A: TensorTupleRepr<1>> tensory_core::concept::task::IsTask for EigRepr<A> {}

impl<A: TensorTupleRepr<1>> EigRepr<A> {
    /// Creates an eigendecomposition representation without validating metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`, and the enclosing tuple's
    /// mappers must match the three output representations.
    pub unsafe fn from_raw_unchecked(a: A, axes_split: EquivGroupedAxes<2>) -> Self {
        Self { a, axes_split }
    }

    /// Decomposes the representation into its input and axis metadata.
    pub fn into_raw(self) -> (A, EquivGroupedAxes<2>) {
        (self.a, self.axes_split)
    }
}

/// Lazy eigendecomposition construction for ordinary tensors.
pub trait TensorEigExt: ToTensor {
    /// Builds an eigendecomposition task with separate IDs for the bonds.
    fn eig_with_more_ids<Q>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvc_leg: <Self::Mapper as AxisMapper>::Id,
        vc_dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, EigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>;

    /// Builds an eigendecomposition task using one shared ID for each bond.
    fn eig<Q>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, EigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone;
}

impl<T: ToTensor> TensorEigExt for T {
    fn eig_with_more_ids<Q>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvc_leg: <Self::Mapper as AxisMapper>::Id,
        vc_dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, EigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        let (raw, [legs]) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.equiv_split(query).map_err(SplittyErr::Split)?;
        let [vc_legs, d_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, v_vd_leg), (1, d_vd_leg)),
                    ((1, d_dvc_leg), (2, vc_dvc_leg)),
                ],
            ))
        }
        .map_err(SplittyErr::Use)?;
        let repr = unsafe { EigRepr::from_raw_unchecked(raw, axes_split) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, [vc_legs, d_legs, v_legs]) })
    }

    fn eig<Q>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<3, EigRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
    {
        self.eig_with_more_ids(query, vd_leg.clone(), vd_leg, dvc_leg.clone(), dvc_leg)
    }
}

/// Runtime-bound eigendecomposition construction and execution.
pub trait BoundTensorEigExt: ToBoundTensorTuple<1> {
    /// Builds and executes an eigendecomposition with separate bond IDs.
    fn eig_with_more_ids<Q, E>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvc_leg: <Self::Mapper as AxisMapper>::Id,
        vc_dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
                3,
                EigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: RuntimeFor<EigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
            3,
            EigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;

    /// Builds and executes an eigendecomposition using shared bond IDs.
    fn eig<Q, E>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
                3,
                EigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<EigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
            3,
            EigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorEigExt for T {
    fn eig_with_more_ids<Q, E>(
        self,
        query: Q,
        v_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_vd_leg: <Self::Mapper as AxisMapper>::Id,
        d_dvc_leg: <Self::Mapper as AxisMapper>::Id,
        vc_dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
                3,
                EigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        Self::Runtime: RuntimeFor<EigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
            3,
            EigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (a, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = a
            .eig_with_more_ids(query, v_vd_leg, d_vd_leg, d_dvc_leg, vc_dvc_leg)
            .map_err(RuntimeErr::Defer)?;
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensorTuple::from_raw(result, runtime))
    }

    fn eig<Q, E>(
        self,
        query: Q,
        vd_leg: <Self::Mapper as AxisMapper>::Id,
        dvc_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            3,
            <<Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
                3,
                EigRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<EigRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<EigRepr<Self::Repr>>>::Mk,
            3,
            EigRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        self.eig_with_more_ids(query, vd_leg.clone(), vd_leg, dvc_leg.clone(), dvc_leg)
    }
}
