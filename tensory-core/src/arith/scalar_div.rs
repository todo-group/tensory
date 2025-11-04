use core::convert::Infallible;
use core::ops::Div;

use crate::{
    bound_tensor::{BoundTensor, Runtime, RuntimeError, ToBoundTensor},
    mapper::AxisMapper,
    repr::TensorRepr,
    tensor::{Tensor, TensorTask},
};

/// Raw context of left scalar division operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait LeftScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

pub struct LeftScalarDiv<A: TensorRepr, M: AxisMapper, E> {
    a: A,
    scalar: E,
    res_broker: M,
}

impl<A: TensorRepr, M: AxisMapper, E, C: LeftScalarDivCtx<A, E>> TensorTask<C>
    for LeftScalarDiv<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.left_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of right scalar division operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait RightScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs right scalar division operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

pub struct TensorRightScalarDiv<A: TensorRepr, M: AxisMapper, E> {
    a: A,
    scalar: E,
    res_broker: M,
}

impl<A: TensorRepr, M: AxisMapper, E, C: RightScalarDivCtx<A, E>> TensorTask<C>
    for TensorRightScalarDiv<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.right_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of scalar division operation. (no left/right)
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait CommutativeScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarDivCtx<A, E>> LeftScalarDivCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarDivCtx<A, E>> RightScalarDivCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}

impl<A: TensorRepr, M: AxisMapper> Tensor<A, M> {
    pub fn left_div<E>(self, lhs: E) -> LeftScalarDiv<A, M, E> {
        let (a, broker) = self.into_raw();
        LeftScalarDiv {
            a,
            scalar: lhs,
            res_broker: broker,
        }
    }
    pub fn right_div<E>(self, rhs: E) -> TensorRightScalarDiv<A, M, E> {
        let (a, broker) = self.into_raw();
        TensorRightScalarDiv {
            a,
            scalar: rhs,
            res_broker: broker,
        }
    }
}

use super::TensorScalar;
use crate::tensor::ToTensor;

macro_rules! impl_scalar_div {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Div<(E,)> for $a
        where
            $a: ToTensor,
        {
            type Output = TensorRightScalarDiv<<Self as ToTensor>::Repr, <Self as ToTensor>::Mapper, E>;
            fn div(self, rhs: (E,)) -> Self::Output {
                self.to_tensor().right_div(rhs.0)
            }
        }
        // impl<$($life,)* A: TensorRepr, M: AxisMapper,E:TensorScalar> Div<$a> for (E,)
        // where
        //     $a: ToTensor,
        // {
        //     type Output = TensorLeftScalarDiv<<$a as ToTensor>::Repr, <$a as ToTensor>::Mapper, E>;
        //     fn div(self, rhs: $a) -> Self::Output {
        //         rhs.to_tensor().left_div(self.0)
        //     }
        // }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E:TensorScalar> Div<E> for $a
        where
            $a: ToTensor,
        {
            type Output = TensorRightScalarDiv<<Self as ToTensor>::Repr, <Self as ToTensor>::Mapper, E>;
            fn div(self, rhs: E) -> Self::Output {
                self.to_tensor().right_div(rhs)
            }
        }

    };
}

impl_scalar_div!(Tensor<A, M>);
impl_scalar_div!(&'a Tensor<A, M>,'a);
impl_scalar_div!(&'a mut Tensor<A, M>,'a);

pub trait LeftScalarDivRuntime<A: TensorRepr, E>: Runtime {
    type Ctx: LeftScalarDivCtx<A, E>;
    fn left_scalar_div_ctx(&self) -> Self::Ctx;
}
pub trait RightScalarDivRuntime<A: TensorRepr, E>: Runtime {
    type Ctx: RightScalarDivCtx<A, E>;
    fn right_scalar_div_ctx(&self) -> Self::Ctx;
}

macro_rules! impl_scalar_div_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Div<(E,)> for $a
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: RightScalarDivRuntime<<$a as ToBoundTensor>::Repr, E>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as RightScalarDivRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarDivCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as RightScalarDivRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarDivCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Err,
                >,
            >;
            fn div(self, rhs: (E,)) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();

                let res = (lhs / rhs)
                    .with(lhs_rt.right_scalar_div_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }


        // impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Div<$a> for (E,)
        // where
        //     $a: ToBoundTensor<Mapper = M, Runtime = RT>,
        //     RT: LeftScalarDivRuntime<<$a as ToBoundTensor>::Repr, E>,
        // {
        //     type Output = Result<
        //         BoundTensor<
        //             <<RT as LeftScalarDivRuntime<
        //                 <$a as ToBoundTensor>::Repr,
        //                 E,
        //             >>::Ctx as LeftScalarDivCtx<
        //                 <$a as ToBoundTensor>::Repr,
        //                 E,
        //             >>::Res,
        //             M,
        //             RT,
        //         >,
        //         RuntimeError<
        //             Infallible,
        //             <<RT as LeftScalarDivRuntime<
        //                 <$a as ToBoundTensor>::Repr,
        //                 E,
        //             >>::Ctx as LeftScalarDivCtx<
        //                 <$a as ToBoundTensor>::Repr,
        //                 E,
        //             >>::Err,
        //         >,
        //     >;
        //     fn div(self, rhs: $a) -> Self::Output {
        //         let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

        //         let res = (self / rhs)
        //             .with(rhs_rt.left_scalar_div_ctx())
        //             .map_err(RuntimeError::Ctx)?;
        //         Ok(BoundTensor::from_raw(res, rhs_rt))
        //     }
        // }

        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E:TensorScalar> Div<E> for $a
        where
            $a: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: RightScalarDivRuntime<<$a as ToBoundTensor>::Repr, E>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as RightScalarDivRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarDivCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    Infallible,
                    <<RT as RightScalarDivRuntime<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Ctx as RightScalarDivCtx<
                        <$a as ToBoundTensor>::Repr,
                        E,
                    >>::Err,
                >,
            >;
            fn div(self, rhs: E) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();

                let res = (lhs / rhs)
                    .with(lhs_rt.right_scalar_div_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_scalar_div_runtime!(BoundTensor<A, M, RT>);
impl_scalar_div_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_div_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
