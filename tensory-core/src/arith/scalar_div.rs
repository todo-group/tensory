use core::{convert::Infallible, ops::Div};

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, RuntimeErr, RuntimeFor},
    },
    mapper::AxisMapper,
    op::{EwiseExt, UnaryEwiseRepr},
    repr::TensorTupleRepr,
    tensor::{
        BoundTensor, Tensor, TensorTuple, TensorTupleContext, ToBoundTensorTuple, ToTensor,
        ToTensorTuple,
    },
};

/// Operation for element-wise left scalar division.
pub struct LeftScalarDivOp<E>(pub E);

/// Lazy representation for a left scalar division operation using the `LeftScalarDivOp`.
pub type LeftScalarDivRepr<const N: usize, A, E> = UnaryEwiseRepr<N, A, LeftScalarDivOp<E>>;

/// Operation for element-wise right scalar division.
pub struct RightScalarDivOp<E>(pub E);

/// Lazy representation for a right scalar division operation using the `RightScalarDivOp`.
pub type RightScalarDivRepr<const N: usize, A, E> = UnaryEwiseRepr<N, A, RightScalarDivOp<E>>;

/// Extension trait for creating scalar division tasks on tensors.
pub trait TensorScalarDivExt<const N: usize, E>: ToTensorTuple<N> {
    /// Creates a left scalar division task.
    fn left_div(self, lhs: E) -> TensorTuple<N, LeftScalarDivRepr<N, Self::Repr, E>, Self::Mapper>
    where
        Self: Sized,
    {
        self.to_tensor_tuple().ewise1(LeftScalarDivOp(lhs))
    }
    /// Creates a right scalar division task.
    fn right_div(self, rhs: E) -> TensorTuple<N, RightScalarDivRepr<N, Self::Repr, E>, Self::Mapper>
    where
        Self: Sized,
    {
        self.to_tensor_tuple().ewise1(RightScalarDivOp(rhs))
    }
}

impl<const N: usize, T: ToTensorTuple<N>, E> TensorScalarDivExt<N, E> for T {}

use super::Scalar;

macro_rules! impl_scalar_div {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Div<(E,)> for $a
        where
            $a: ToTensor,
        {
            type Output = Tensor<RightScalarDivRepr<1,<$a as ToTensor>::Repr, E>, <$a as ToTensor>::Mapper>;
            fn div(self, rhs: (E,)) -> Self::Output {
                self.to_tensor().ewise1(RightScalarDivOp(rhs.0))
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E: Scalar> Div<E> for $a
        where
            $a: ToTensor,
        {
            type Output = Tensor<RightScalarDivRepr<1,<$a as ToTensor>::Repr, E>, <$a as ToTensor>::Mapper>;
            fn div(self, rhs: E) -> Self::Output {
                self.to_tensor().ewise1(RightScalarDivOp(rhs))
            }
        }

    };
}

impl_scalar_div!(Tensor<A, M>);
impl_scalar_div!(&'a Tensor<A, M>,'a);
impl_scalar_div!(&'a mut Tensor<A, M>,'a);

macro_rules! impl_scalar_div_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, RT: IsRuntime, E, Err>
            Div<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarDivRepr<1, <$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarDivRepr<1, <$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarDivRepr<1, <$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarDivRepr<1, <$a as ToBoundTensorTuple<1>>::Repr, E>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn div(self, rhs: (E,)) -> Self::Output {
                let (lhs, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(lhs / rhs)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }

        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, E: Scalar, Err>
            Div<E> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarDivRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarDivRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarDivRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarDivRepr<1,<$a as ToBoundTensorTuple<1>>::Repr, E>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn div(self, rhs: E) -> Self::Output {
                let (lhs, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(lhs / rhs)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }
    };
}

impl_scalar_div_runtime!(BoundTensor<A, M, RT>);
impl_scalar_div_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_div_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
