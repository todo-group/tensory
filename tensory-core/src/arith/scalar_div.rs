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
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait LeftScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

/// Intermediate task struct for left scalar division operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TensorLeftScalarDiv<A: TensorRepr, M: AxisMapper, E> {
    a: A,
    scalar: E,
    res_mapper: M,
}

impl<A: TensorRepr, M: AxisMapper, E, C: LeftScalarDivCtx<A, E>> TensorTask<C>
    for TensorLeftScalarDiv<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.left_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mapper) })
    }
}

/// Raw context of right scalar division operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait RightScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs right scalar division operation on the tensor `a`.
    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}

/// Intermediate task struct for right scalar division operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TensorRightScalarDiv<A: TensorRepr, M: AxisMapper, E> {
    a: A,
    scalar: E,
    res_mapper: M,
}

impl<A: TensorRepr, M: AxisMapper, E, C: RightScalarDivCtx<A, E>> TensorTask<C>
    for TensorRightScalarDiv<A, M, E>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = ctx.right_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mapper) })
    }
}

/// Raw context of commutative scalar division operation. (no left/right)
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait CommutativeScalarDivCtx<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
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

/// Extension trait for left/right scalar division operation on tensors.
pub trait TensorScalarDivExt<E> {
    /// The type of the tensor representation.
    type A: TensorRepr;
    /// The type of the axis mapper.
    type M: AxisMapper;
    /// Creates a left scalar division task.
    fn left_div(self, lhs: E) -> TensorLeftScalarDiv<Self::A, Self::M, E>;
    /// Creates a right scalar division task.
    fn right_div(self, rhs: E) -> TensorRightScalarDiv<Self::A, Self::M, E>;
}

impl<T: ToTensor, E> TensorScalarDivExt<E> for T {
    type A = T::Repr;
    type M = T::Mapper;

    fn left_div(self, lhs: E) -> TensorLeftScalarDiv<Self::A, Self::M, E> {
        let (a, mapper) = self.to_tensor().into_raw();
        TensorLeftScalarDiv {
            a,
            scalar: lhs,
            res_mapper: mapper,
        }
    }
    fn right_div(self, rhs: E) -> TensorRightScalarDiv<Self::A, Self::M, E> {
        let (a, mapper) = self.to_tensor().into_raw();
        TensorRightScalarDiv {
            a,
            scalar: rhs,
            res_mapper: mapper,
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

/// Runtime trait for left scalar division operation.
pub trait LeftScalarDivRuntime<A: TensorRepr, E>: Runtime {
    /// The context type.
    type Ctx: LeftScalarDivCtx<A, E>;
    /// Returns the context.
    fn left_scalar_div_ctx(&self) -> Self::Ctx;
}

/// Runtime trait for right scalar division operation.
pub trait RightScalarDivRuntime<A: TensorRepr, E>: Runtime {
    /// The context type.
    type Ctx: RightScalarDivCtx<A, E>;
    /// Returns the context.
    fn right_scalar_div_ctx(&self) -> Self::Ctx;
}

/// Runtime trait for commutative scalar division operation. (no left/right)
pub trait CommutativeScalarDivRuntime<A: TensorRepr, E>: Runtime {
    /// The context type.
    type Ctx: CommutativeScalarDivCtx<A, E>;
    /// Returns the context.
    fn scalar_div_ctx(&self) -> Self::Ctx;
}
impl<T: CommutativeScalarDivRuntime<A, E>, A: TensorRepr, E> LeftScalarDivRuntime<A, E> for T {
    type Ctx = T::Ctx;
    fn left_scalar_div_ctx(&self) -> Self::Ctx {
        self.scalar_div_ctx()
    }
}
impl<T: CommutativeScalarDivRuntime<A, E>, A: TensorRepr, E> RightScalarDivRuntime<A, E> for T {
    type Ctx = T::Ctx;
    fn right_scalar_div_ctx(&self) -> Self::Ctx {
        self.scalar_div_ctx()
    }
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
