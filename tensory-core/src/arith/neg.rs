use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, RuntimeErr, RuntimeFor},
    },
    mapper::AxisMapper,
    op::{EwiseExt, UnaryEwiseRepr},
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

use core::{convert::Infallible, ops::Neg};

/*
/// Raw context of negation operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait NegCtx<A: TensorTupleRepr<1>> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs negation operation on the tensor `a`.
    fn negate(self, a: A) -> Result<Self::Res, Self::Err>;
}

*/

/// Operation for element-wise negation.
pub struct NegOp;

/// Lazy representation for a negation operation using the `NegOp`.
pub type NegRepr<const N: usize, A> = UnaryEwiseRepr<N, A, NegOp>;

macro_rules! impl_neg {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper> Neg for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<NegRepr<1, <$a as ToTensor>::Repr>, M>;
            fn neg(self) -> Self::Output {
                let a = ToTensor::to_tensor(self);
                a.ewise1(NegOp)
            }
        }
    };
}

impl_neg!(Tensor<A, M>);
impl_neg!(&'a Tensor<A, M>,'a);
impl_neg!(&'a mut Tensor<A, M>,'a);

// Runtime-bound implementations for negation.
macro_rules! impl_neg_runtime {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, Err> Neg for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<NegRepr<1,<$a as ToBoundTensorTuple<1>>::Repr>, M>,
            >,
            <RT as RuntimeFor<
                Tensor<NegRepr<1,<$a as ToBoundTensorTuple<1>>::Repr>, M>,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                NegRepr<1,<$a as ToBoundTensorTuple<1>>::Repr>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output =
            Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        NegRepr<1,<$a as ToBoundTensorTuple<1>>::Repr>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn neg(self) -> Self::Output {
                let (a, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(-a)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }
    };
}

impl_neg_runtime!(BoundTensor<A, M,RT>);
impl_neg_runtime!(&'a BoundTensor<A, M,RT>,'a);
impl_neg_runtime!(&'a mut BoundTensor<A, M,RT>,'a);
