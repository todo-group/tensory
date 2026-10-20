use core::convert::Infallible;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeErr, RuntimeFor},
    },
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a conjugation operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ConjRepr<A: TensorTupleRepr<1>> {
    a: A,
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<1> for ConjRepr<A> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>> tensory_core::concept::task::IsTask for ConjRepr<A> {}

impl<A: TensorTupleRepr<1>> ConjRepr<A> {
    /// Creates a conjugation representation without validating its input.
    ///
    /// # Safety
    ///
    /// The context must return a representation with the same axis structure
    /// as `a`.
    pub unsafe fn from_raw_unchecked(a: A) -> Self {
        Self { a }
    }

    /// Decomposes the representation into its input representation.
    pub fn into_raw(self) -> A {
        self.a
    }
}

/// Lazy conjugation construction for ordinary tensors.
pub trait TensorConjExt: ToTensor {
    /// The output tensor type produced by conjugation.
    type Output;

    /// Builds a lazy conjugation task.
    fn conj(self) -> Self::Output;
}

impl<T: ToTensor> TensorConjExt for T {
    type Output = Tensor<ConjRepr<T::Repr>, T::Mapper>;

    fn conj(self) -> Self::Output {
        let (a, [mapper]) = self.to_tensor().into_raw();
        let task = unsafe { ConjRepr::from_raw_unchecked(a) };
        unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
    }
}

/// Runtime-bound conjugation construction and execution.
pub trait BoundTensorConjExt: ToBoundTensorTuple<1> {
    /// Builds and executes a conjugation task.
    fn conj<E>(
        self,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Mk,
                1,
                ConjRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<Infallible, E>,
    >
    where
        Self::Runtime: RuntimeFor<ConjRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Mk,
            1,
            ConjRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorConjExt for T {
    fn conj<E>(
        self,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Ctx as TensorTupleContext<
                <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Mk,
                1,
                ConjRepr<Self::Repr>,
                Self::Mapper,
            >>::Repr,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeErr<Infallible, E>,
    >
    where
        Self::Runtime: RuntimeFor<ConjRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Ctx: TensorTupleContext<
            <Self::Runtime as RuntimeFor<ConjRepr<Self::Repr>>>::Mk,
            1,
            ConjRepr<Self::Repr>,
            Self::Mapper,
            CType = Resulting<Raw, E>,
        >,
    {
        let (tensor, runtime) = self.to_bound_tensor_tuple().into_raw();
        let (repr, [mapper]) = tensor.into_raw();
        let task = unsafe {
            Tensor::from_raw_unchecked(ConjRepr::from_raw_unchecked(repr), [mapper])
        };
        let result = runtime
            .ctx()
            .execute(task)
            .map_err(RuntimeErr::Execute)?;
        Ok(BoundTensor::from_raw(result, runtime))
    }
}
