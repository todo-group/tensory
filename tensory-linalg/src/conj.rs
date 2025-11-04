use tensory_core::{
    bound_tensor::BoundTensor,
    mapper::AxisMapper,
    repr::{AsViewMutRepr, AsViewRepr, TensorRepr},
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of conjugation operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait ConjCtx<A: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs conjugation operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn conjugate(self, a: A) -> Result<Self::Res, Self::Err>;
}

pub struct TensorConj<A: TensorRepr, B: AxisMapper> {
    a: A,
    res_mgr: B,
}

impl<A: TensorRepr, M: AxisMapper, C: ConjCtx<A>> TensorTask<C> for TensorConj<A, M> {
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;

        let aconj = ctx.conjugate(a)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mgr) })
    }
}

pub trait TensorConjExt {
    type Output;
    fn conj(self) -> Self::Output;
}

impl<T: ToTensor> TensorConjExt for T {
    type Output = TensorConj<T::Repr, T::Mapper>;
    fn conj(self) -> Self::Output {
        let (a, mgr) = self.to_tensor().into_raw();
        TensorConj { a, res_mgr: mgr }
    }
}

// impl<'rt, A: TensorRepr, B: AxisMapper, RT> TensorWithRuntime<A, B, RT>
// where
//     &'rt RT: ConjugationContext<A>,
// {
//     pub fn conj(
//         self,
//     ) -> Result<
//         TensorWithRuntime<'rt, <&'rt RT as ConjugationContext<A>>::Res, B, RT>,
//         <&'rt RT as ConjugationContext<A>>::Err,
//     > {
//         let (t, rt) = self.into_raw();

//         let t = t.conj().with(rt)?;

//         Ok(TensorWithRuntime::from_raw(t, rt))
//     }
// }
