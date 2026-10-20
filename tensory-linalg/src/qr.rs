use alloc::vec;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeErr, RuntimeFor},
    },
    mapper::{AxisMapper, DecompConf, DecompGroupedMapper, GroupMapper, GroupedAxes, SplittyErr},
    repr::TensorTupleRepr,
    tensor::{BoundTensorTuple, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a QR operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QrRepr<A: TensorTupleRepr<1>> {
    a: A,
    axes_split: GroupedAxes<2>,
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<2> for QrRepr<A> {
    fn naxes_array(&self) -> [usize; 2] {
        let (_, [q_axes, r_axes]) = self.axes_split.clone().into_raw();
        [q_axes.len() + 1, r_axes.len() + 1]
    }
}

impl<A: TensorTupleRepr<1>> tensory_core::concept::task::IsTask for QrRepr<A> {}

impl<A: TensorTupleRepr<1>> QrRepr<A> {
    /// Creates a QR representation without validating its metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`, and the enclosing tuple's
    /// mappers must match the two output representations.
    pub unsafe fn from_raw_unchecked(a: A, axes_split: GroupedAxes<2>) -> Self {
        Self { a, axes_split }
    }

    /// Decomposes the representation into its input and axis metadata.
    pub fn into_raw(self) -> (A, GroupedAxes<2>) {
        (self.a, self.axes_split)
    }
}

/// Lazy QR construction for ordinary tensors.
pub trait TensorQrExt: ToTensor {
    /// Builds a QR task with separate IDs for the two bonds.
    fn qr_with_more_ids<Q>(
        self,
        query: Q,
        q_qr_leg: <Self::Mapper as AxisMapper>::Id,
        r_qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, QrRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>;

    /// Builds a QR task using one shared ID for both bonds.
    fn qr<Q>(
        self,
        query: Q,
        qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, QrRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone;
}

impl<T: ToTensor> TensorQrExt for T {
    fn qr_with_more_ids<Q>(
        self,
        query: Q,
        q_qr_leg: <Self::Mapper as AxisMapper>::Id,
        r_qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, QrRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
    {
        let (raw, [legs]) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.split(query).map_err(SplittyErr::Split)?;
        let [q_legs, r_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 1],
                vec![((0, q_qr_leg), (1, r_qr_leg))],
            ))
        }
        .map_err(SplittyErr::Use)?;
        let repr = unsafe { QrRepr::from_raw_unchecked(raw, axes_split) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, [q_legs, r_legs]) })
    }

    fn qr<Q>(
        self,
        query: Q,
        qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorTuple<2, QrRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as GroupMapper<2, Q>>::Err,
            <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
    {
        self.qr_with_more_ids(query, qr_leg.clone(), qr_leg)
    }
}

/// Runtime-bound QR construction and execution.
pub trait BoundTensorQrExt: ToBoundTensorTuple<1> {
    /// Builds and executes a QR with separate IDs for the two bonds.
    fn qr_with_more_ids<Q, E>(
        self,
        query: Q,
        q_qr_leg: <Self::Mapper as AxisMapper>::Id,
        r_qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
                2,
                QrRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        Self::Runtime: RuntimeFor<QrRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
            2,
            QrRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;

    /// Builds and executes a QR using one shared ID for both bonds.
    fn qr<Q, E>(
        self,
        query: Q,
        qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
                2,
                QrRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<QrRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
            2,
            QrRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorQrExt for T {
    fn qr_with_more_ids<Q, E>(
        self,
        query: Q,
        q_qr_leg: <Self::Mapper as AxisMapper>::Id,
        r_qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
                2,
                QrRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        Self::Runtime: RuntimeFor<QrRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
            2,
            QrRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (a, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = a
            .qr_with_more_ids(query, q_qr_leg, r_qr_leg)
            .map_err(RuntimeErr::Defer)?;
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensorTuple::from_raw(result, runtime))
    }

    fn qr<Q, E>(
        self,
        query: Q,
        qr_leg: <Self::Mapper as AxisMapper>::Id,
    ) -> Result<
        BoundTensorTuple<
            2,
            <<Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
                2,
                QrRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as GroupMapper<2, Q>>::Err,
                <<Self::Mapper as GroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 2>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: GroupMapper<2, Q>,
        <Self::Mapper as GroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 2>,
        <Self::Mapper as AxisMapper>::Id: Clone,
        Self::Runtime: RuntimeFor<QrRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<QrRepr<Self::Repr>>>::Mk,
            2,
            QrRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        self.qr_with_more_ids(query, qr_leg.clone(), qr_leg)
    }
}
