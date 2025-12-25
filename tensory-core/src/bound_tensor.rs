//! Layer 3 tensor concept: layer 2 tensor bound with runtime.

use thiserror::Error;

use crate::{
    mapper::{AxisMapper, ReplaceMapper},
    repr::{AsViewMutRepr, AsViewRepr, TensorRepr},
    tensor::{Tensor, TensorExt},
};

/// A tensor bound with a runtime.
///
///
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct BoundTensor<R: TensorRepr, M: AxisMapper, RT: Runtime> {
    tensor: Tensor<R, M>,
    runtime: RT,
}

pub unsafe trait Runtime: Clone + Eq {}

impl<R: TensorRepr, M: AxisMapper> Tensor<R, M> {
    /// Binds the tensor with a runtime, producing a runtime-bound tensor.
    pub fn bind<RT: Runtime>(self, runtime: RT) -> BoundTensor<R, M, RT> {
        BoundTensor::from_raw(self, runtime)
    }
}
impl<R: TensorRepr, M: AxisMapper, RT: Runtime> BoundTensor<R, M, RT> {
    /// Creates a runtime-bound tensor from a tensor and a runtime.
    pub fn from_raw(tensor: Tensor<R, M>, runtime: RT) -> Self {
        Self { tensor, runtime }
    }
    /// Decomposes the runtime-bound tensor into a tensor and a runtime.
    pub fn into_raw(self) -> (Tensor<R, M>, RT) {
        (self.tensor, self.runtime)
    }
    /// Unbinds the runtime from a runtime-bound tensor, returning the underlying tensor.
    pub fn unbind(self) -> Tensor<R, M> {
        self.tensor
    }
    /// Get the immutable reference to the tensor.
    pub fn tensor(&self) -> &Tensor<R, M> {
        &self.tensor
    }
    /// Get the mutable reference to the tensor.
    pub fn tensor_mut(&mut self) -> &mut Tensor<R, M> {
        &mut self.tensor
    }
    /// Get the immutable reference to the runtime.
    pub fn runtime(&self) -> &RT {
        &self.runtime
    }
    /// Get the mutable reference to the runtime.
    pub fn runtime_mut(&mut self) -> &mut RT {
        &mut self.runtime
    }
}

/// General utility trait for bound tensor operations.
pub trait BoundTensorExt: ToBoundTensor {
    /// Replace a ID of a leg of the tensor.
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>;
}

impl<T: ToBoundTensor> BoundTensorExt for T {
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>,
    {
        let (t, rt) = self.to_bound_tensor().into_raw();
        t.replace_leg(query).map(|t| t.bind(rt))
    }
}

mod private {
    use crate::{
        bound_tensor::{BoundTensor, Runtime},
        mapper::AxisMapper,
        repr::TensorRepr,
    };
    pub trait ToBoundTensorSealed {}
    impl<T: TensorRepr, M: AxisMapper, RT: Runtime> ToBoundTensorSealed for BoundTensor<T, M, RT> {}
    impl<T: TensorRepr, M: AxisMapper, RT: Runtime> ToBoundTensorSealed for &BoundTensor<T, M, RT> {}
    impl<T: TensorRepr, M: AxisMapper, RT: Runtime> ToBoundTensorSealed for &mut BoundTensor<T, M, RT> {}
}

/// Conversion trait to TensorWithRuntime.
///
/// This trait is sealed, and implemented for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime. &TensorWithRuntime and &mut TensorWithRuntime will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime; this is required because `&` operator is not overloadable.
pub trait ToBoundTensor: private::ToBoundTensorSealed {
    /// The representation type of the resulting tensor.
    type Repr: TensorRepr;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// The runtime type of the resulting tensor.
    type Runtime: Runtime;
    /// Converts itself to a tensor.
    fn to_bound_tensor(self) -> BoundTensor<Self::Repr, Self::Mapper, Self::Runtime>;
}

impl<T: TensorRepr, M: AxisMapper, RT: Runtime> ToBoundTensor for BoundTensor<T, M, RT> {
    type Repr = T;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor(self) -> BoundTensor<Self::Repr, Self::Mapper, Self::Runtime> {
        self
    }
}
impl<'a, T: AsViewRepr<'a>, M: AxisMapper + Clone, RT: Runtime> ToBoundTensor
    for &'a BoundTensor<T, M, RT>
{
    type Repr = T::View;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor(self) -> BoundTensor<Self::Repr, Self::Mapper, Self::Runtime> {
        self.tensor().view().bind(self.runtime().clone())
    }
}
impl<'a, T: AsViewMutRepr<'a>, M: AxisMapper + Clone, RT: Runtime> ToBoundTensor
    for &'a mut BoundTensor<T, M, RT>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor(self) -> BoundTensor<Self::Repr, Self::Mapper, Self::Runtime> {
        let runtime = self.runtime().clone();
        self.tensor_mut().view_mut().bind(runtime)
    }
}

/// Error type for operations involving runtime context.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Error)]
pub enum RuntimeErr<AE, CE> {
    /// Runtime mismatch.
    #[error("Runtime mismatch")]
    Runtime,
    /// Delegated axis error.
    #[error("Axis error: {0}")]
    Axis(AE),
    /// Delegated context error.
    #[error("Context error: {0}")]
    Ctx(CE),
}
