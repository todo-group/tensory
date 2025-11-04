use core::{convert::Infallible, ops::Mul};

use crate::{
    bound_tensor::{BoundTensor, Runtime, RuntimeError, ToBoundTensor},
    mapper::AxisMapper,
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of left scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait LeftScalarMulCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar multiplication operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

pub struct TensorLeftScalarMul<A: TensorRepr, M: AxisMapper, E> {
    a: A,
    scalar: E,
    res_broker: M,
}

// impl<A: TensorRepr, M: AxisMapper, E> TensorLeftScalarMul<A, M, E> {
//     // pub fn new(a: Tensor<LA, A>) -> Self {
//     //     let (raw, legs) = a.into_raw();
//     //     Self { a: raw, legs }
//     // }
// }

impl<A: TensorRepr, M: AxisMapper, E, C: LeftScalarMulCtx<A, E>> TensorTask<C>
    for TensorLeftScalarMul<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.left_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of right scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait RightScalarMulCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs right scalar multiplication operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

pub struct TensorRightScalarMul<A: TensorRepr, B: AxisMapper, E> {
    a: A,
    scalar: E,
    res_broker: B,
}

// impl<A: TensorRepr, B: AxisMapper, E> TensorRightScalarMul<A, B, E> {
//     // pub fn new(a: Tensor<LA, A>) -> Self {
//     //     let (raw, legs) = a.into_raw();
//     //     Self { a: raw, legs }
//     // }
// }

impl<A: TensorRepr, M: AxisMapper, E, C: RightScalarMulCtx<A, E>> TensorTask<C>
    for TensorRightScalarMul<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.right_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of scalar multiplication operation. (no left/right)
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait CommutativeScalarMulCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar multiplication operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarMulCtx<A, E>> LeftScalarMulCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarMulCtx<A, E>> RightScalarMulCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}

trait TensorScalarMulExt<E> {
    type A: TensorRepr;
    type M: AxisMapper;
    fn left_mul(self, lhs: E) -> TensorLeftScalarMul<Self::A, Self::M, E>;
    fn right_mul(self, rhs: E) -> TensorRightScalarMul<Self::A, Self::M, E>;
}

impl<T: ToTensor, E> TensorScalarMulExt<E> for T {
    type A = T::Repr;
    type M = T::Mapper;

    fn left_mul(self, lhs: E) -> TensorLeftScalarMul<Self::A, Self::M, E> {
        let (a, broker) = self.to_tensor().into_raw();
        TensorLeftScalarMul {
            a,
            scalar: lhs,
            res_broker: broker,
        }
    }
    fn right_mul(self, rhs: E) -> TensorRightScalarMul<Self::A, Self::M, E> {
        let (a, broker) = self.to_tensor().into_raw();
        TensorRightScalarMul {
            a,
            scalar: rhs,
            res_broker: broker,
        }
    }
}

use super::TensorScalar;

macro_rules! impl_scalar_mul {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Mul<(E,)> for $a
        where
            $a: ToTensor,
        {
            type Output = TensorRightScalarMul<<Self as ToTensor>::Repr, <Self as ToTensor>::Mapper, E>;
            fn mul(self, rhs: (E,)) -> Self::Output {
                self.to_tensor().right_mul(rhs.0)
            }
        }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Mul<$a> for (E,)
        where
            $a: ToTensor,
        {
            type Output = TensorLeftScalarMul<<$a as ToTensor>::Repr, <$a as ToTensor>::Mapper, E>;
            fn mul(self, rhs: $a) -> Self::Output {
                rhs.to_tensor().left_mul(self.0)
            }
        }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E:TensorScalar> Mul<E> for $a
        where
            $a: ToTensor,
        {
            type Output = TensorRightScalarMul<<Self as ToTensor>::Repr, <Self as ToTensor>::Mapper, E>;
            fn mul(self, rhs: E) -> Self::Output {
                self.to_tensor().right_mul(rhs)
            }
        }

    };
}

impl_scalar_mul!(Tensor<A, M>);
impl_scalar_mul!(&'a Tensor<A, M>,'a);
impl_scalar_mul!(&'a mut Tensor<A, M>,'a);

pub trait LeftScalarMulRuntime<A: TensorRepr, E>: Runtime {
    type Ctx: LeftScalarMulCtx<A, E>;
    fn left_scalar_mul_ctx(&self) -> Self::Ctx;
}
pub trait RightScalarMulRuntime<A: TensorRepr, E>: Runtime {
    type Ctx: RightScalarMulCtx<A, E>;
    fn right_scalar_mul_ctx(&self) -> Self::Ctx;
}

macro_rules! impl_scalar_mul_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Mul<(E,)> for $a
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: RightScalarMulRuntime<<$a as ToBoundTensor>::Repr, E>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as RightScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as RightScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Err,
                >,
            >;
            fn mul(self, rhs: (E,)) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();

                let res = (lhs * rhs)
                    .with(lhs_rt.right_scalar_mul_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }


        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Mul<$a> for (E,)
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: LeftScalarMulRuntime<<$a as ToBoundTensor>::Repr, E>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as LeftScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as LeftScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as LeftScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as LeftScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Err,
                >,
            >;
            fn mul(self, rhs: $a) -> Self::Output {
                let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

                let res = (self * rhs)
                    .with(rhs_rt.left_scalar_mul_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, rhs_rt))
            }
        }

        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E:TensorScalar> Mul<E> for $a
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: RightScalarMulRuntime<<$a as ToBoundTensor>::Repr, E>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as RightScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as RightScalarMulRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarMulCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Err,
                >,
            >;
            fn mul(self, rhs: E) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();

                let res = (lhs * rhs)
                    .with(lhs_rt.right_scalar_mul_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_scalar_mul_runtime!(BoundTensor<A, M, RT>);
impl_scalar_mul_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_mul_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
