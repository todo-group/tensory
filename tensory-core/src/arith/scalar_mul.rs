use core::{convert::Infallible, ops::Mul};

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::AxisMapper,
    op::{EwiseExt, UnaryEwiseRepr},
    repr::TensorTupleRepr,
    tensor::{
        BoundTensor, Tensor, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor,
        ToTensorTuple,
    },
};

pub struct LeftScalarMulOp<E>(E);

pub type LeftScalarMulRepr<const N: usize, A, E> = UnaryEwiseRepr<N, A, LeftScalarMulOp<E>>;

pub struct RightScalarMulOp<E>(E);

pub type RightScalarMulRepr<const N: usize, A, E> = UnaryEwiseRepr<N, A, RightScalarMulOp<E>>;

/// Extension trait for left/right scalar multiplication operation on tensors.
pub trait TensorScalarMulExt<const N: usize, E>: ToTensorTuple<N> {
    /// Creates a left scalar multiplication task.
    fn left_mul(self, lhs: E) -> TensorTuple<N, LeftScalarMulRepr<N, Self::Repr, E>, Self::Mapper>
    where
        Self: Sized,
    {
        self.to_tensor_tuple().ewise1(LeftScalarMulOp(lhs))
    }
    /// Creates a right scalar multiplication task.
    fn right_mul(self, rhs: E) -> TensorTuple<N, RightScalarMulRepr<N, Self::Repr, E>, Self::Mapper>
    where
        Self: Sized,
    {
        self.to_tensor_tuple().ewise1(RightScalarMulOp(rhs))
    }
}

impl<const N: usize, T: ToTensorTuple<N>, E> TensorScalarMulExt<N, E> for T {}

use super::Scalar;

macro_rules! impl_scalar_mul {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Mul<(E,)> for $a
        where
            $a: ToTensor,
        {
            type Output = Tensor<RightScalarMulRepr<1,<$a as ToTensor>::Repr, E>, <$a as ToTensor>::Mapper>;
            fn mul(self, rhs: (E,)) -> Self::Output {
                self.to_tensor().right_mul(rhs.0)
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Mul<$a> for (E,)
        where
            $a: ToTensor
        {
            type Output = Tensor<LeftScalarMulRepr<1,<$a as ToTensor>::Repr, E>, <$a as ToTensor>::Mapper>;
            fn mul(self, rhs: $a) -> Self::Output {
                rhs.to_tensor().left_mul(self.0)
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper,E: Scalar> Mul<E> for $a
        where
            $a: ToTensor,
        {
            type Output = Tensor<RightScalarMulRepr<1, <$a as ToTensor>::Repr, E>, <$a as ToTensor>::Mapper>;
            fn mul(self, rhs: E) -> Self::Output {
                self.to_tensor().right_mul(rhs)
            }
        }
    };
}

impl_scalar_mul!(Tensor<A, M>);
impl_scalar_mul!(&'a Tensor<A, M>,'a);
impl_scalar_mul!(&'a mut Tensor<A, M>,'a);

macro_rules! impl_scalar_mul_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, RT: IsRuntime, E, Err>
            Mul<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn mul(self, rhs: (E,)) -> Self::Output {
                let (lhs, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(lhs * rhs)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }

        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, E, Err>
            Mul<$a> for (E,)
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    LeftScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    LeftScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                LeftScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        LeftScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn mul(self, rhs: $a) -> Self::Output {
                let (rhs, rt) = rhs.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(self * rhs)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }

        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, E: Scalar, Err>
            Mul<E> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarMulRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn mul(self, rhs: E) -> Self::Output {
                let (lhs, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(lhs * rhs)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }
    };
}

impl_scalar_mul_runtime!(BoundTensor<A, M, RT>);
impl_scalar_mul_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_mul_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
