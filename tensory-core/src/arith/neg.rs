use crate::{
    bound_tensor::{BoundTensorTuple, Runtime, RuntimeErr, ToBoundTensorTuple},
    mapper::AxisMapper,
    repr::TensorRepr,
    task::{Context, TaskDelegate, TaskHolder},
    tensor::{Tensor, ToTensorTuple},
};

use core::convert::Infallible;
use core::ops::Neg;

// /// Raw context of negation operation.
// ///
// /// # Safety
// ///
// /// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
// pub unsafe trait NegCtx<A: TensorRepr> {
//     /// The type of the result tensor representation.
//     type Res: TensorRepr;
//     /// The type of the error returned by the context. (considered as internal error)
//     type Err;

//     /// Performs negation operation on the tensor `a`.
//     fn negate(self, a: A) -> Result<Self::Res, Self::Err>;
// }

/// Intermediate task struct for negation operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NegRepr<A: TensorRepr> {
    a: A,
}

impl<A: TensorRepr> NegRepr<A> {
    pub fn from_raw(a: A) -> Self {
        Self { a }
    }

    pub fn into_raw(self) -> A {
        self.a
    }
}

unsafe impl<A: TensorRepr> TensorRepr for NegRepr<A> {
    fn naxes(&self) -> usize {
        self.a.naxes()
    }
}

pub unsafe trait TensorNegContext<Mk, A: TensorRepr, O>: Context<Mk, NegRepr<A>, O> {
    // type Ctx: Context<Mk, AddRepr<L, R>, O>;
    // fn add_ctx(&self) -> Self::Ctx;
}

impl<A: TensorRepr, M: AxisMapper> TaskHolder<NegRepr<A>> for Tensor<NegRepr<A>, M> {}

impl<A: TensorRepr, M: AxisMapper, Mk, Ctx: TensorNegContext<Mk, A, O>, O: TensorRepr>
    TaskDelegate<NegRepr<A>, O, Mk, Ctx> for Tensor<NegRepr<A>, M>
{
    type Output = Tensor<O, M>;
    fn with(self, ctx: Ctx) -> Self::Output {
        let (repr, mapper) = self.into_raw();
        let output = ctx.execute(repr);

        unsafe { Tensor::from_raw_unchecked(output, mapper) }
    }
}
impl<
    A: TensorRepr,
    M: AxisMapper,
    Mk,
    Ctx: TensorNegContext<Mk, A, Result<Ores, Oerr>>,
    Ores: TensorRepr,
    Oerr,
> TaskDelegate<NegRepr<A>, Result<Ores, Oerr>, Mk, Ctx> for Tensor<NegRepr<A>, M>
{
    type Output = Result<Tensor<Ores, M>, Oerr>;
    fn with(self, ctx: Ctx) -> Self::Output {
        let (repr, mapper) = self.into_raw();
        let output = ctx.execute(repr)?;

        Ok(unsafe { Tensor::from_raw_unchecked(output, mapper) })
    }
}

macro_rules! impl_neg {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper> Neg for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<NegRepr<<Self as ToTensor>::Repr>, M>;
            fn neg(self) -> Self::Output {
                let (a, mapper)=self.to_tensor().into_raw();
                unsafe { Tensor::from_raw_unchecked(NegRepr { a }, mapper) }
            }
        }
    };
}

impl_neg!(Tensor<A, M>);
impl_neg!(&'a Tensor<A, M>,'a);
impl_neg!(&'a mut Tensor<A, M>,'a);

// /// Runtime trait for negation operation.
// pub trait NegRuntime<A: TensorRepr>: Runtime {
//     /// The context type.
//     type Ctx: NegCtx<A>;
//     /// Returns the context.
//     fn neg_ctx(&self) -> Self::Ctx;
// }

// macro_rules! impl_neg_runtime {
//     ($a:ty $(,$life:lifetime)* ) => {
//         impl<$($life,)* A: TensorRepr, M: AxisMapper,RT:Runtime> Neg for $a
//         where
//             $a: ToBoundTensor<Mapper = M, Runtime = RT>,
//             RT: NegRuntime<<$a as ToBoundTensor>::Repr>,
//         {
//             type Output =
//             Result<
//                 BoundTensor<
//                     <<RT as NegRuntime<
//                         <$a as ToBoundTensor>::Repr
//                     >>::Ctx as NegCtx<
//                         <$a as ToBoundTensor>::Repr
//                     >>::Res,
//                     M,
//                     RT,
//                 >,
//                 RuntimeErr<
//                     Infallible,
//                     <<RT as NegRuntime<
//                         <$a as ToBoundTensor>::Repr
//                     >>::Ctx as NegCtx<
//                         <$a as ToBoundTensor>::Repr
//                     >>::Err,
//                 >,
//             >;
//             fn neg(self) -> Self::Output {
//                 let (a, a_rt) = self.to_bound_tensor().into_raw();
//                 let res = (-a)
//                     .with(a_rt.neg_ctx())
//                     .map_err(RuntimeErr::Ctx)?;
//                 Ok(BoundTensor::from_raw(res, a_rt))
//             }
//         }
//     };
// }

// impl_neg_runtime!(BoundTensor<A, M,RT>);
// impl_neg_runtime!(&'a BoundTensor<A, M,RT>,'a);
// impl_neg_runtime!(&'a mut BoundTensor<A, M,RT>,'a);
