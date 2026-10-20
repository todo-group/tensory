use core::convert::Infallible;

use tensory_core::{
    concept::task::{Context, IsTask, RuntimeErr, RuntimeFor},
    repr::TensorTupleRepr,
    tensor::{ToBoundTensorTuple, ToTensor},
};

/// Lazy representation of a norm operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NormRepr<A: TensorTupleRepr<1>> {
    a: A,
}

impl<A: TensorTupleRepr<1>> IsTask for NormRepr<A> {}

impl<A: TensorTupleRepr<1>> NormRepr<A> {
    /// Creates a norm representation.
    pub fn from_raw(a: A) -> Self {
        Self { a }
    }

    /// Decomposes the representation into its input.
    pub fn into_raw(self) -> A {
        self.a
    }
}

/// Lazy norm construction for ordinary tensors.
pub trait TensorNormExt: ToTensor {
    /// Builds a lazy norm task.
    fn norm(self) -> NormRepr<Self::Repr>;
}

impl<T: ToTensor> TensorNormExt for T {
    fn norm(self) -> NormRepr<Self::Repr> {
        let (repr, _) = self.to_tensor().into_raw();
        NormRepr::from_raw(repr)
    }
}

/// Runtime-bound norm construction and execution.
pub trait BoundTensorNormExt: ToBoundTensorTuple<1> {
    /// Builds and executes a norm task.
    fn norm<R, E>(self) -> Result<R, RuntimeErr<Infallible, E>>
    where
        Self::Runtime: RuntimeFor<NormRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<NormRepr<Self::Repr>>>::Ctx: Context<
            <Self::Runtime as RuntimeFor<NormRepr<Self::Repr>>>::Mk,
            NormRepr<Self::Repr>,
            Output = Result<R, E>,
        >;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorNormExt for T {
    fn norm<R, E>(self) -> Result<R, RuntimeErr<Infallible, E>>
    where
        Self::Runtime: RuntimeFor<NormRepr<Self::Repr>>,
        <Self::Runtime as RuntimeFor<NormRepr<Self::Repr>>>::Ctx: Context<
            <Self::Runtime as RuntimeFor<NormRepr<Self::Repr>>>::Mk,
            NormRepr<Self::Repr>,
            Output = Result<R, E>,
        >,
    {
        let (tensor, runtime) = self.to_bound_tensor_tuple().into_raw();
        let (repr, _) = tensor.into_raw();
        let result = runtime
            .ctx()
            .execute(NormRepr::from_raw(repr))
            .map_err(RuntimeErr::Execute)?;
        Ok(result)
    }
}
