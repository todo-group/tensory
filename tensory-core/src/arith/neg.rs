use crate::{
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::AxisMapper,
    repr::{TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

use core::ops::Neg;

/// Intermediate task representation for negation operation.
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

unsafe impl<A: TensorRepr> TensorTupleRepr<1> for NegRepr<A> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr> IsTask for NegRepr<A> {}

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_neg {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper> Neg for $a
        where
            $a: ToTensorTuple<1,Mapper = M>,
        {
            type Output = Tensor<NegRepr<<Self as ToTensorTuple<1>>::Repr>, M>;
            fn neg(self) -> Self::Output {
                let (a, [mapper]) = self.to_tensor_tuple().into_raw();
                unsafe { Tensor::from_raw_unchecked(NegRepr::from_raw(a), [mapper]) }
            }
        }
    };
}
impl_neg!(Tensor<A, M>);
impl_neg!(&'a Tensor<A, M>,'a);
impl_neg!(&'a mut Tensor<A, M>,'a);

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_neg_runtime {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,RT:Runtime> Neg for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>
            >>::Container;

            fn neg(self) -> Self::Output {
                let (a, a_rt) = self.to_bound_tensor_tuple().into_raw();
                let res = a_rt.ctx().execute(-a);
                <RT::Ctx as TensorContext<RT::Mk, 1, NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, a_rt)
                })
            }
        }
    };
}
impl_neg_runtime!(BoundTensor<A, M, RT>);
impl_neg_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_neg_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
