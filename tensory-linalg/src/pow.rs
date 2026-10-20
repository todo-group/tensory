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

/// Lazy representation of a matrix power operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PowRepr<A: TensorTupleRepr<1>, P> {
    a: A,
    power: P,
    axes_split: EquivGroupedAxes<2>,
}

unsafe impl<A: TensorTupleRepr<1>, P> TensorTupleRepr<1> for PowRepr<A, P> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>, P> tensory_core::concept::task::IsTask for PowRepr<A, P> {}

impl<A: TensorTupleRepr<1>, P> PowRepr<A, P> {
    /// Creates a power representation without validating metadata.
    ///
    /// # Safety
    ///
    /// `axes_split` must describe all axes of `a`.
    pub unsafe fn from_raw_unchecked(a: A, power: P, axes_split: EquivGroupedAxes<2>) -> Self {
        Self {
            a,
            power,
            axes_split,
        }
    }

    /// Decomposes the representation into its input, power, and axis metadata.
    pub fn into_raw(self) -> (A, P, EquivGroupedAxes<2>) {
        (self.a, self.power, self.axes_split)
    }
}

/// Lazy matrix power construction for ordinary tensors.
pub trait TensorPowExt: ToTensor {
    /// Builds a lazy power task.
    fn pow<P, Q>(
        self,
        power: P,
        query: Q,
    ) -> Result<
        Tensor<PowRepr<Self::Repr, P>, Self::Mapper>,
        SplittyErr<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>;
}

impl<T: ToTensor> TensorPowExt for T {
    fn pow<P, Q>(
        self,
        power: P,
        query: Q,
    ) -> Result<
        Tensor<PowRepr<Self::Repr, P>, Self::Mapper>,
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
        let repr = unsafe { PowRepr::from_raw_unchecked(raw, power, axes_split) };
        Ok(unsafe { Tensor::from_raw_unchecked(repr, [mapper]) })
    }
}

/// Runtime-bound power construction and execution.
pub trait BoundTensorPowExt: ToBoundTensorTuple<1> {
    /// Builds and executes a power task.
    fn pow<P, Q, E>(
        self,
        power: P,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Mk,
                1,
                PowRepr<Self::Repr, P>,
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
        Self::Runtime: RuntimeFor<PowRepr<Self::Repr, P>>,
        <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Mk,
            1,
            PowRepr<Self::Repr, P>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorPowExt for T {
    fn pow<P, Q, E>(
        self,
        power: P,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Mk,
                1,
                PowRepr<Self::Repr, P>,
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
        Self::Runtime: RuntimeFor<PowRepr<Self::Repr, P>>,
        <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<PowRepr<Self::Repr, P>>>::Mk,
            1,
            PowRepr<Self::Repr, P>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (tensor, runtime) = self.to_bound_tensor_tuple().into_raw();
        let task = tensor.pow(power, query).map_err(RuntimeErr::Defer)?;
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensor::from_raw(result, runtime))
    }
}
