use tensory_core::{
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
        let (a, _broker) = self.to_tensor().into_raw();
        TensorNorm { a }
    }
}
