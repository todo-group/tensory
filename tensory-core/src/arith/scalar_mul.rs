use core::{convert::Infallible, ops::Mul};

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::AxisMapper,
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/*
/// Raw context of left scalar multiplication operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait LeftScalarMulCtx<A: TensorTupleRepr<1>, E> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar multiplication operation on the tensor `a`.
    fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
*/

/// Lazy representation for a left scalar multiplication operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LeftScalarMulRepr<A: TensorTupleRepr<1>, E> {
    a: A,
    scalar: E,
}

impl<A: TensorTupleRepr<1>, E> LeftScalarMulRepr<A, E> {
    /// Creates a representation for a left scalar multiplication.
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }

    /// Creates a representation without checking its input.
    ///
    /// # Safety
    ///
    /// `a` must be a valid tensor representation.
    pub unsafe fn from_raw_unchecked(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }

    /// Decomposes the representation into its input and scalar.
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorTupleRepr<1>, E> TensorTupleRepr<1> for LeftScalarMulRepr<A, E> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>, E> IsTask for LeftScalarMulRepr<A, E> {}

/*
/// Raw context of right scalar multiplication operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait RightScalarMulCtx<A: TensorTupleRepr<1>, E> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs right scalar multiplication operation on the tensor `a`.
    fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
*/

/// Lazy representation for a right scalar multiplication operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RightScalarMulRepr<A: TensorTupleRepr<1>, E> {
    a: A,
    scalar: E,
}

impl<A: TensorTupleRepr<1>, E> RightScalarMulRepr<A, E> {
    /// Creates a representation for a right scalar multiplication.
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }

    /// Creates a representation without checking its input.
    ///
    /// # Safety
    ///
    /// `a` must be a valid tensor representation.
    pub unsafe fn from_raw_unchecked(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }

    /// Decomposes the representation into its input and scalar.
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorTupleRepr<1>, E> TensorTupleRepr<1> for RightScalarMulRepr<A, E> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>, E> IsTask for RightScalarMulRepr<A, E> {}

/*
/// Raw context of scalar multiplication operation. (no left/right)
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait CommutativeScalarMulCtx<A: TensorTupleRepr<1>, E>:
    LeftScalarMulCtx<A, E>
    + RightScalarMulCtx<
        A,
        E,
        Res = <Self as LeftScalarMulCtx<A, E>>::Res,
        Err = <Self as LeftScalarMulCtx<A, E>>::Err,
    > + Sized
{
    // /// The type of the result tensor representation.
    // type Res: TensorRepr;
    // /// The type of the error returned by the context. (considered as internal error)
    // type Err;

    /// Performs left scalar multiplication operation on the tensor `a`.
    fn scalar_mul(
        self,
        a: A,
        scalar: E,
    ) -> Result<<Self as LeftScalarMulCtx<A, E>>::Res, <Self as LeftScalarMulCtx<A, E>>::Err> {
        self.left_scalar_mul(a, scalar)
    }
}

// pub unsafe trait ScalarMulCtxDerive<A: TensorRepr, E> {
//     type Res: TensorRepr;
//     type Err;

//     fn scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
// }

// unsafe impl<A: TensorRepr, E, C: ScalarMulCtxDerive<A, E>> LeftScalarMulCtx<A, E> for C {
//     type Res = <C as ScalarMulCtxDerive<A, E>>::Res;
//     type Err = <C as ScalarMulCtxDerive<A, E>>::Err;

//     fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
//         self.scalar_mul(a, scalar)
//     }
// }
// unsafe impl<A: TensorRepr, E, C: ScalarMulCtxDerive<A, E>> RightScalarMulCtx<A, E> for C {
//     type Res = <C as ScalarMulCtxDerive<A, E>>::Res;
//     type Err = <C as ScalarMulCtxDerive<A, E>>::Err;

//     fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
//         self.scalar_mul(a, scalar)
//     }
// }
// unsafe impl<A: TensorRepr, E, C: ScalarMulCtxDerive<A, E>> CommutativeScalarMulCtx<A, E> for C {}
*/

/// Extension trait for left/right scalar multiplication operation on tensors.
pub trait TensorScalarMulExt<E> {
    /// The type of the tensor representation.
    type A: TensorTupleRepr<1>;
    /// Creates a left scalar multiplication task.
    fn left_mul(self, lhs: E) -> LeftScalarMulRepr<Self::A, E>;
    /// Creates a right scalar multiplication task.
    fn right_mul(self, rhs: E) -> RightScalarMulRepr<Self::A, E>;
}

impl<T: ToTensor, E> TensorScalarMulExt<E> for T {
    type A = T::Repr;

    fn left_mul(self, lhs: E) -> LeftScalarMulRepr<Self::A, E> {
        let (a, [_mapper]) = self.to_tensor().into_raw();
        LeftScalarMulRepr { a, scalar: lhs }
    }
    fn right_mul(self, rhs: E) -> RightScalarMulRepr<Self::A, E> {
        let (a, [_mapper]) = self.to_tensor().into_raw();
        RightScalarMulRepr { a, scalar: rhs }
    }
}

use super::Scalar;

macro_rules! impl_scalar_mul {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Mul<(E,)> for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<RightScalarMulRepr<<$a as ToTensor>::Repr, E>, M>;
            fn mul(self, rhs: (E,)) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(self).into_raw();
                let task = RightScalarMulRepr::from_raw(a, rhs.0);
                unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Mul<$a> for (E,)
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<LeftScalarMulRepr<<$a as ToTensor>::Repr, E>, M>;
            fn mul(self, rhs: $a) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(rhs).into_raw();
                let task = LeftScalarMulRepr::from_raw(a, self.0);
                unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E: Scalar> Mul<E> for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<RightScalarMulRepr<<$a as ToTensor>::Repr, E>, M>;
            fn mul(self, rhs: E) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(self).into_raw();
                let task = RightScalarMulRepr::from_raw(a, rhs);
                unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
            }
        }

    };
}

impl_scalar_mul!(Tensor<A, M>);
impl_scalar_mul!(&'a Tensor<A, M>,'a);
impl_scalar_mul!(&'a mut Tensor<A, M>,'a);

macro_rules! impl_scalar_mul_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, E, Err>
            Mul<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
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
                    LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
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
                    RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
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
