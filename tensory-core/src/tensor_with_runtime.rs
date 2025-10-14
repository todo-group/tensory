//! Layer 3 tensor concept: layer 2 tensor bound with runtime.

use thiserror::Error;

use crate::{
    mapper::AxisMapper,
    repr::{AsViewMutRepr, AsViewRepr, TensorRepr},
    tensor::Tensor,
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct TensorWithRuntime<R: TensorRepr, M: AxisMapper, RT> {
    tensor: Tensor<R, M>,
    runtime: RT,
}

impl<R: TensorRepr, M: AxisMapper> Tensor<R, M> {
    /// Binds the tensor with a runtime, producing a `TensorWithRuntime`.
    pub fn bind<RT>(self, runtime: RT) -> TensorWithRuntime<R, M, RT> {
        TensorWithRuntime::from_raw(self, runtime)
    }
}
impl<R: TensorRepr, M: AxisMapper, RT> TensorWithRuntime<R, M, RT> {
    /// Creates a `TensorWithRuntime` from `Tensor` and runtime.
    pub fn from_raw(tensor: Tensor<R, M>, runtime: RT) -> Self {
        Self { tensor, runtime }
    }
    /// Decomposes the `TensorWithRuntime` into `Tensor` and runtime.
    pub fn into_raw(self) -> (Tensor<R, M>, RT) {
        (self.tensor, self.runtime)
    }
    /// Unbinds the tensor from its runtime, returning the underlying `Tensor`.
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

mod private {
    use crate::{mapper::AxisMapper, repr::TensorRepr, tensor_with_runtime::TensorWithRuntime};
    pub trait ToTensorWithRuntimeSealed {}
    impl<T: TensorRepr, M: AxisMapper, RT> ToTensorWithRuntimeSealed for TensorWithRuntime<T, M, RT> {}
    impl<T: TensorRepr, M: AxisMapper, RT> ToTensorWithRuntimeSealed for &TensorWithRuntime<T, M, RT> {}
    impl<T: TensorRepr, M: AxisMapper, RT> ToTensorWithRuntimeSealed
        for &mut TensorWithRuntime<T, M, RT>
    {
    }
}

/// Conversion trait to TensorWithRuntime.
///
/// This trait is sealed, and implemented for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime. &TensorWithRuntime and &mut TensorWithRuntime will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime; this is required because `&` operator is not overloadable.
pub trait ToTensorWithRuntime: private::ToTensorWithRuntimeSealed {
    /// The representation type of the resulting tensor.
    type Repr: TensorRepr;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// The runtime type of the resulting tensor.
    type Runtime: Copy + Eq;
    /// Converts itself to a tensor.
    fn to_tensor_with_runtime(self) -> TensorWithRuntime<Self::Repr, Self::Mapper, Self::Runtime>;
}

impl<T: TensorRepr, M: AxisMapper, RT: Copy + Eq> ToTensorWithRuntime
    for TensorWithRuntime<T, M, RT>
{
    type Repr = T;
    type Mapper = M;
    type Runtime = RT;
    fn to_tensor_with_runtime(self) -> TensorWithRuntime<Self::Repr, Self::Mapper, Self::Runtime> {
        self
    }
}
impl<'a, T: AsViewRepr<'a>, M: AxisMapper + Clone, RT: Copy + Eq> ToTensorWithRuntime
    for &'a TensorWithRuntime<T, M, RT>
{
    type Repr = T::View;
    type Mapper = M;
    type Runtime = RT;
    fn to_tensor_with_runtime(self) -> TensorWithRuntime<Self::Repr, Self::Mapper, Self::Runtime> {
        self.tensor().view().bind(*self.runtime())
    }
}
impl<'a, T: AsViewMutRepr<'a>, M: AxisMapper + Clone, RT: Copy + Eq> ToTensorWithRuntime
    for &'a mut TensorWithRuntime<T, M, RT>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    type Runtime = RT;
    fn to_tensor_with_runtime(self) -> TensorWithRuntime<Self::Repr, Self::Mapper, Self::Runtime> {
        let runtime = *self.runtime();
        self.tensor_mut().view_mut().bind(runtime)
    }
}

// pub enum ContractExecutorError<Err> {
//     MismatchedExecutors,
//     Internal(Err),
// }

// trait Executor {
//     fn exec_eq(&self, other: &Self) -> bool;
// }

// impl<Id, L: TensorRepr, Ex: Executor> Tensor<Id, L, Ex> {
//     pub fn contract<R: TensorRepr>(self,rhs: Tensor<Id, R, Ex>) -> Result<Tensor<Id, Ex::Res, Ex>, ContractExecutorError<Ex::Err>> where Ex: ContractionContext<L, R> {
//         self.executor.exec_eq(&rhs.executor).then(()).ok_or(ContractExecutorError::MismatchedExecutors).and()

//         //map(|_|TensorMul::new(self, rhs).by(context).map_err(ContractExecutorError::Internal))
//         todo!()
//     }
// }

// impl<Id, L: TensorRepr, R: TensorRepr, Ex: ContractionContext<L, R>> Mul<Tensor<Id, R, Ex>>
//     for Tensor<Id, L, Ex>
// {
//     type Output = TensorMul<Id, <Ex as ContractionContext<L, R>>::Res, Ex>;

//     fn mul(self, rhs: Tensor<Id, R, Ex>) -> Self::Output {
//         if self.executor != rhs.executor {
//             panic!("Cannot multiply tensors with different executors");
//         }

//         TensorMul::new(self, rhs)
//     }
// }

// impl<Id, A: TensorRepr, Ex: SvdContext<A>> Tensor<Id, A, Ex> {
//     pub fn svd<'a>(self, u_legs: LegRefSet<'a, Id>) ->  {
//         TensorSvd::new(self, u_legs)
//     }
// }

/// Error type for operations involving runtime context.
#[derive(Error, Debug)]
pub enum RuntimeError<AE, CE> {
    #[error("Runtime mismatch")]
    Runtime,
    #[error("Axis error: {0}")]
    Axis(AE),
    #[error("Context error: {0}")]
    Ctx(CE),
}
