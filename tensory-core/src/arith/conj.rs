use crate::tensor::{
    AsViewMutRepr, AsViewRepr, Tensor, TensorBroker, TensorRepr, TensorWithRuntime,
};

/// Raw context of conjugation operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait ConjugationContext<A: TensorRepr> {
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

pub struct TensorConj<A: TensorRepr, B: TensorBroker> {
    a: A,
    res_mgr: B,
}

impl<A: TensorRepr, B: TensorBroker> TensorConj<A, B> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: ConjugationContext<A>>(self, context: C) -> Result<Tensor<C::Res, B>, C::Err> {
        let a = self.a;

        let aconj = context.conjugate(a)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mgr) })
    }
}

pub trait Conj {
    type Output;
    fn conj(self) -> Self::Output;
}

impl<A: TensorRepr, B: TensorBroker> Conj for Tensor<A, B> {
    type Output = TensorConj<A, B>;
    fn conj(self) -> Self::Output {
        let (a, mgr) = self.into_raw();
        TensorConj { a, res_mgr: mgr }
    }
}
impl<'a, A: TensorRepr + AsViewRepr<'a>, B: TensorBroker + Clone> Conj for &'a Tensor<A, B> {
    type Output = TensorConj<A::View, B>;
    fn conj(self) -> TensorConj<A::View, B> {
        let (a, mgr) = self.view().into_raw();
        TensorConj { a, res_mgr: mgr }
    }
}
impl<'a, A: TensorRepr + AsViewMutRepr<'a>, B: TensorBroker + Clone> Conj for &'a mut Tensor<A, B> {
    type Output = TensorConj<A::ViewMut, B>;
    fn conj(self) -> TensorConj<A::ViewMut, B> {
        let (a, mgr) = self.view_mut().into_raw();
        TensorConj { a, res_mgr: mgr }
    }
}

impl<'rt, A: TensorRepr, B: TensorBroker, RT> TensorWithRuntime<'rt, A, B, RT>
where
    &'rt RT: ConjugationContext<A>,
{
    pub fn conj(
        self,
    ) -> Result<
        TensorWithRuntime<'rt, <&'rt RT as ConjugationContext<A>>::Res, B, RT>,
        <&'rt RT as ConjugationContext<A>>::Err,
    > {
        let (t, rt) = self.into_raw();

        let t = t.conj().with(rt)?;

        Ok(TensorWithRuntime::from_raw(t, rt))
    }
}
