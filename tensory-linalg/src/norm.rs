use core::convert::Infallible;

use tensory_core::{
    bound_tensor::{Runtime, RuntimeError, ToBoundTensor},
    repr::TensorRepr,
    tensor::{TensorTask, ToTensor},
};

/// Raw context of norm operation.
pub trait NormCtx<A: TensorRepr> {
    /// The type of the result tensor representation.
    type Res;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs norm operation on the tensor `a`.
    fn norm(self, a: A) -> Result<Self::Res, Self::Err>;
}

pub struct TensorNorm<A: TensorRepr> {
    a: A,
}

impl<A: TensorRepr, C: NormCtx<A>> TensorTask<C> for TensorNorm<A> {
    type Output = Result<C::Res, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        ctx.norm(a)
    }
}

pub trait TensorNormExt<A: TensorRepr>: Sized {
    fn norm(self) -> TensorNorm<A>;
}

impl<T: ToTensor> TensorNormExt<T::Repr> for T {
    fn norm(self) -> TensorNorm<T::Repr> {
        let (a, _) = self.to_tensor().into_raw();
        TensorNorm { a }
    }
}

pub trait NormRuntime<A: TensorRepr>: Runtime {
    type Ctx: NormCtx<A>;
    fn norm_ctx(&self) -> Self::Ctx;
}

pub trait BoundTensorNormExt: ToBoundTensor {
    fn norm(
        self,
    ) -> Result<
        <<Self::Runtime as NormRuntime<Self::Repr>>::Ctx as NormCtx<Self::Repr>>::Res,
        RuntimeError<
            Infallible,
            <<Self::Runtime as NormRuntime<Self::Repr>>::Ctx as NormCtx<Self::Repr>>::Err,
        >,
    >
    where
        Self::Runtime: NormRuntime<Self::Repr>;
}

impl<T: ToBoundTensor> BoundTensorNormExt for T {
    fn norm(
        self,
    ) -> Result<
        <<Self::Runtime as NormRuntime<Self::Repr>>::Ctx as NormCtx<Self::Repr>>::Res,
        RuntimeError<
            Infallible,
            <<Self::Runtime as NormRuntime<Self::Repr>>::Ctx as NormCtx<Self::Repr>>::Err,
        >,
    >
    where
        T::Runtime: NormRuntime<Self::Repr>,
    {
        let (a, rt) = self.to_bound_tensor().into_raw();
        a.norm().with(rt.norm_ctx()).map_err(RuntimeError::Ctx)
    }
}
