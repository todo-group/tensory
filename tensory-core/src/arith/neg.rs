use crate::{
    bound_tensor::{BoundTensor, Runtime, RuntimeError, ToBoundTensor},
    mapper::AxisMapper,
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

use core::convert::Infallible;
use core::ops::Neg;

/// Raw context of negation operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait NegCtx<A: TensorRepr> {
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

pub struct TensorNeg<A: TensorRepr, M: AxisMapper> {
    a: A,
    res_broker: M,
}

// pub fn new(a: Tensor<LA, A>) -> Self {
//     let (raw, legs) = a.into_raw();
//     Self { a: raw, legs }
// }

impl<A: TensorRepr, M: AxisMapper, C: NegCtx<A>> TensorTask<C> for TensorNeg<A, M> {
    type Output = Result<Tensor<C::Res, M>, C::Err>;
    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;

        let aneg = ctx.negate(a)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aneg, self.res_broker) })
    }
}

macro_rules! impl_neg {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper> Neg for $a
        where
            $a: ToTensor,
        {
            type Output = TensorNeg<<Self as ToTensor>::Repr, <Self as ToTensor>::Mapper>;
            fn neg(self) -> Self::Output {
                let (a, mgr)=self.to_tensor().into_raw();
                TensorNeg { a, res_broker: mgr }
            }
        }
    };
}

impl_neg!(Tensor<A, M>);
impl_neg!(&'a Tensor<A, M>,'a);
impl_neg!(&'a mut Tensor<A, M>,'a);

pub trait NegRuntime<A: TensorRepr>: Runtime {
    type Ctx: NegCtx<A>;
    fn neg_ctx(&self) -> Self::Ctx;
}

macro_rules! impl_neg_runtime {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,RT:Runtime> Neg for $a
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: NegRuntime<<$a as ToBoundTensor>::Repr>,
        {
            type Output =
            Result<
                BoundTensor<
                    <<RT as NegRuntime<
                        <$a as ToBoundTensor>::Repr
                    >>::Ctx as NegCtx<
                        <$a as ToBoundTensor>::Repr
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as NegRuntime<
                        <$a as ToBoundTensor>::Repr
                    >>::Ctx as NegCtx<
                        <$a as ToBoundTensor>::Repr
                    >>::Err,
                >,
            >;
            fn neg(self) -> Self::Output {
                let (a, a_rt) = self.to_bound_tensor().into_raw();
                let res = (-a)
                    .with(a_rt.neg_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, a_rt))
            }
        }
    };
}

impl_neg_runtime!(BoundTensor<A, M,RT>);
impl_neg_runtime!(&'a BoundTensor<A, M,RT>,'a);
impl_neg_runtime!(&'a mut BoundTensor<A, M,RT>,'a);
