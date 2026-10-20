use alloc::vec;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeErr, RuntimeFor},
    },
    mapper::{DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes, SplittyErr},
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a matrix exponential operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ExpRepr<A: TensorTupleRepr<1>> {
    a: A,
    axes_split: EquivGroupedAxes<2>,
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<1> for ExpRepr<A> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>> tensory_core::concept::task::IsTask for ExpRepr<A> {}

impl<A: TensorTupleRepr<1>> ExpRepr<A> {
    /// Creates an exponential representation without validating metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`.
    pub unsafe fn from_raw_unchecked(a: A, axes_split: EquivGroupedAxes<2>) -> Self {
        Self { a, axes_split }
    }

    /// Decomposes the representation into its input and axis metadata.
    pub fn into_raw(self) -> (A, EquivGroupedAxes<2>) {
        (self.a, self.axes_split)
    }
}

/// Lazy matrix exponential construction for ordinary tensors.
pub trait TensorExpExt: ToTensor {
    /// Builds a lazy exponential task.
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        Tensor<ExpRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>;
}

impl<T: ToTensor> TensorExpExt for T {
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        Tensor<ExpRepr<Self::Repr>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
    {
        let (raw, [legs]) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.equiv_split(query).map_err(SplittyErr::Split)?;
        let [mapper] = unsafe { grouped.decomp(DecompConf::from_raw_unchecked([0, 0], vec![])) }
            .map_err(SplittyErr::Use)?;
        let repr = unsafe { ExpRepr::from_raw_unchecked(raw, axes_split) };
        Ok(unsafe { Tensor::from_raw_unchecked(repr, [mapper]) })
    }
}

/// Runtime-bound exponential construction and execution.
pub trait BoundTensorExpExt: ToBoundTensorTuple<1> {
    /// Builds and executes an exponential task.
    fn exp<Q, E>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Mk,
                1,
                ExpRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: RuntimeFor<ExpRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Mk,
            1,
            ExpRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorExpExt for T {
    fn exp<Q, E>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Mk,
                1,
                ExpRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<
            SplittyErr<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            E,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: RuntimeFor<ExpRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<ExpRepr<Self::Repr>>>::Mk,
            1,
            ExpRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (tensor, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = tensor.exp(query).map_err(RuntimeErr::Defer)?;
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensor::from_raw(result, runtime))
    }
}
