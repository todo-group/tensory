//! Layer 3 tensor concept: layer 2 tensor bound with runtime.

use thiserror::Error;

use crate::{
    mapper::{AxisMapper, ReplaceMapper},
    repr::{AsViewMutRepr, AsViewRepr, TensorTupleRepr},
    task::Context,
    tensor::{TensorExt, TensorTuple},
};

/// A tensor bound with a runtime.
///
///
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct BoundTensorTuple<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime> {
    tensor: TensorTuple<N, R, M>,
    runtime: RT,
}

pub unsafe trait Runtime: Clone + Eq {}
pub trait RuntimeImpl<T>: Runtime {
    /// context marker type
    type Mk;
    /// The context type.
    type Ctx: Context<Self::Mk, T>;
    /// Returns the context.
    fn ctx(&self) -> Self::Ctx;
}

impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper> TensorTuple<N, R, M> {
    /// Binds the tensor with a runtime, producing a runtime-bound tensor.
    pub fn bind<RT: Runtime>(self, runtime: RT) -> BoundTensorTuple<N, R, M, RT> {
        BoundTensorTuple::from_raw(self, runtime)
    }
}
impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime>
    BoundTensorTuple<N, R, M, RT>
{
    /// Creates a runtime-bound tensor from a tensor and a runtime.
    pub fn from_raw(tensor: TensorTuple<N, R, M>, runtime: RT) -> Self {
        Self { tensor, runtime }
    }
    /// Decomposes the runtime-bound tensor into a tensor and a runtime.
    pub fn into_raw(self) -> (TensorTuple<N, R, M>, RT) {
        (self.tensor, self.runtime)
    }
    /// Unbinds the runtime from a runtime-bound tensor, returning the underlying tensor.
    pub fn unbind(self) -> TensorTuple<N, R, M> {
        self.tensor
    }
    /// Get the immutable reference to the tensor.
    pub fn tensor(&self) -> &TensorTuple<N, R, M> {
        &self.tensor
    }
    /// Get the mutable reference to the tensor.
    pub fn tensor_mut(&mut self) -> &mut TensorTuple<N, R, M> {
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

mod private {
    use crate::{
        bound_tensor::{BoundTensorTuple, Runtime},
        mapper::AxisMapper,
        repr::TensorTupleRepr,
    };
    pub trait ToBoundTensorSealed<const N: usize> {}
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime> ToBoundTensorSealed<N>
        for BoundTensorTuple<N, T, M, RT>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime> ToBoundTensorSealed<N>
        for &BoundTensorTuple<N, T, M, RT>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime> ToBoundTensorSealed<N>
        for &mut BoundTensorTuple<N, T, M, RT>
    {
    }
}

/// Conversion trait to TensorWithRuntime.
///
/// This trait is sealed, and implemented for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime. &TensorWithRuntime and &mut TensorWithRuntime will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime; this is required because `&` operator is not overloadable.
pub trait ToBoundTensorTuple<const N: usize>: private::ToBoundTensorSealed<N> {
    /// The representation type of the resulting tensor.
    type Repr: TensorTupleRepr<N>;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// The runtime type of the resulting tensor.
    type Runtime: Runtime;
    /// Converts itself to a tensor.
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime>;
}

impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: Runtime> ToBoundTensorTuple<N>
    for BoundTensorTuple<N, T, M, RT>
{
    type Repr = T;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        self
    }
}
impl<'a, const N: usize, T: AsViewRepr<'a, N>, M: AxisMapper + Clone, RT: Runtime>
    ToBoundTensorTuple<N> for &'a BoundTensorTuple<N, T, M, RT>
{
    type Repr = T::View;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        self.tensor().view().bind(self.runtime().clone())
    }
}
impl<'a, const N: usize, T: AsViewMutRepr<'a, N>, M: AxisMapper + Clone, RT: Runtime>
    ToBoundTensorTuple<N> for &'a mut BoundTensorTuple<N, T, M, RT>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        let runtime = self.runtime().clone();
        self.tensor_mut().view_mut().bind(runtime)
    }
}

pub type BoundTensor<R, M, RT> = BoundTensorTuple<1, R, M, RT>;

/// General utility trait for bound tensor operations.
pub trait BoundTensorExt: ToBoundTensorTuple<1> {
    /// Replace a ID of a leg of the tensor.
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensorTuple<1, Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorExt for T {
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensorTuple<1, Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>,
    {
        let (t, rt) = self.to_bound_tensor_tuple().into_raw();
        t.replace_leg(query).map(|t| t.bind(rt))
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
