use core::{convert::Infallible, ops::Div};

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
/// Raw context of left scalar division operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait LeftScalarDivCtx<A: TensorTupleRepr<1>, E> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
*/

/// Lazy representation for a left scalar division operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LeftScalarDivRepr<A: TensorTupleRepr<1>, E> {
    a: A,
    scalar: E,
}

impl<A: TensorTupleRepr<1>, E> LeftScalarDivRepr<A, E> {
    /// Creates a representation for a left scalar division.
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

unsafe impl<A: TensorTupleRepr<1>, E> TensorTupleRepr<1> for LeftScalarDivRepr<A, E> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>, E> IsTask for LeftScalarDivRepr<A, E> {}

/*
/// Raw context of right scalar division operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait RightScalarDivCtx<A: TensorTupleRepr<1>, E> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs right scalar division operation on the tensor `a`.
    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
*/

/// Lazy representation for a right scalar division operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RightScalarDivRepr<A: TensorTupleRepr<1>, E> {
    a: A,
    scalar: E,
}

impl<A: TensorTupleRepr<1>, E> RightScalarDivRepr<A, E> {
    /// Creates a representation for a right scalar division.
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

unsafe impl<A: TensorTupleRepr<1>, E> TensorTupleRepr<1> for RightScalarDivRepr<A, E> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>, E> IsTask for RightScalarDivRepr<A, E> {}

/*
/// Raw context of commutative scalar division operation. (no left/right)
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait CommutativeScalarDivCtx<A: TensorTupleRepr<1>, E> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    fn scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
unsafe impl<A: TensorTupleRepr<1>, E, C: CommutativeScalarDivCtx<A, E>> LeftScalarDivCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}
unsafe impl<A: TensorTupleRepr<1>, E, C: CommutativeScalarDivCtx<A, E>> RightScalarDivCtx<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}
*/

/// Extension trait for left/right scalar division operation on tensors.
pub trait TensorScalarDivExt<E> {
    /// The type of the tensor representation.
    type A: TensorTupleRepr<1>;
    /// Creates a left scalar division task.
    fn left_div(self, lhs: E) -> LeftScalarDivRepr<Self::A, E>;
    /// Creates a right scalar division task.
    fn right_div(self, rhs: E) -> RightScalarDivRepr<Self::A, E>;
}

impl<T: ToTensor, E> TensorScalarDivExt<E> for T {
    type A = T::Repr;

    fn left_div(self, lhs: E) -> LeftScalarDivRepr<Self::A, E> {
        let (a, [_mapper]) = self.to_tensor().into_raw();
        LeftScalarDivRepr { a, scalar: lhs }
    }
    fn right_div(self, rhs: E) -> RightScalarDivRepr<Self::A, E> {
        let (a, [_mapper]) = self.to_tensor().into_raw();
        RightScalarDivRepr { a, scalar: rhs }
    }
}

use super::Scalar;

macro_rules! impl_scalar_div {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E> Div<(E,)> for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<RightScalarDivRepr<<$a as ToTensor>::Repr, E>, M>;
            fn div(self, rhs: (E,)) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(self).into_raw();
                let task = RightScalarDivRepr::from_raw(a, rhs.0);
                unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
            }
        }
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper, E: Scalar> Div<E> for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<RightScalarDivRepr<<$a as ToTensor>::Repr, E>, M>;
            fn div(self, rhs: E) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(self).into_raw();
                let task = RightScalarDivRepr::from_raw(a, rhs);
                unsafe { Tensor::from_raw_unchecked(task, [mapper]) }
            }
        }

    };
}

impl_scalar_div!(Tensor<A, M>);
impl_scalar_div!(&'a Tensor<A, M>,'a);
impl_scalar_div!(&'a mut Tensor<A, M>,'a);

macro_rules! impl_scalar_div_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, E, Err>
            Div<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
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
                    RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>,
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
