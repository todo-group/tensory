use crate::tensor::{Tensor, TensorBroker, TensorRepr};

/// Raw context of negation operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait NegationContext<A: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs negation operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn negate(self, a: A) -> Result<Self::Res, Self::Err>;
}

pub struct TensorNeg<M: TensorBroker, A: TensorRepr> {
    a: A,
    res_mgr: M,
}

impl<M: TensorBroker, A: TensorRepr> TensorNeg<M, A> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: NegationContext<A>>(self, context: C) -> Result<Tensor<M, C::Res>, C::Err> {
        let a = self.a;

        let aneg = context.negate(a)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aneg, self.res_mgr) })
    }
}

impl<M: TensorBroker, T: TensorRepr> Tensor<M, T> {
    pub fn neg(self) -> TensorNeg<M, T> {
        let (a, mgr) = self.into_raw();
        TensorNeg { a, res_mgr: mgr }
    }
}
