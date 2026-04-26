use core::ops::Mul;

use crate::{
    arith::TensorScalar,
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::AxisMapper,
    repr::{ReprContext, TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

/// Intermediate task representation for left scalar multiplication (scalar * tensor) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LeftScalarMulRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> LeftScalarMulRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for LeftScalarMulRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for LeftScalarMulRepr<A, E> {}

/// Intermediate task representation for right scalar multiplication (tensor * scalar) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RightScalarMulRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> RightScalarMulRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for RightScalarMulRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for RightScalarMulRepr<A, E> {}

/// Intermediate task representation for commutative scalar multiplication (tensor * scalar) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CommutativeScalarMulRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> CommutativeScalarMulRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for CommutativeScalarMulRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for CommutativeScalarMulRepr<A, E> {}

impl<A: TensorRepr, E, Mk, C: Context<Mk, CommutativeScalarMulRepr<A, E>>>
    Context<Mk, LeftScalarMulRepr<A, E>> for C
{
    type Output = <C as Context<Mk, CommutativeScalarMulRepr<A, E>>>::Output;

    fn execute(self, task: LeftScalarMulRepr<A, E>) -> Self::Output {
        self.execute(CommutativeScalarMulRepr {
            a: task.a,
            scalar: task.scalar,
        })
    }
}
unsafe impl<A: TensorRepr, E, Mk, C: ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>
    ReprContext<Mk, 1, LeftScalarMulRepr<A, E>> for C
{
    type Repr = <C as ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>::Repr;
    type CType = <C as ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>::CType;
}

impl<A: TensorRepr, E, Mk, C: Context<Mk, CommutativeScalarMulRepr<A, E>>>
    Context<Mk, RightScalarMulRepr<A, E>> for C
{
    type Output = <C as Context<Mk, CommutativeScalarMulRepr<A, E>>>::Output;

    fn execute(self, task: RightScalarMulRepr<A, E>) -> Self::Output {
        self.execute(CommutativeScalarMulRepr {
            a: task.a,
            scalar: task.scalar,
        })
    }
}
unsafe impl<A: TensorRepr, E, Mk, C: ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>
    ReprContext<Mk, 1, RightScalarMulRepr<A, E>> for C
{
    type Repr = <C as ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>::Repr;
    type CType = <C as ReprContext<Mk, 1, CommutativeScalarMulRepr<A, E>>>::CType;
}

/// Extension trait for left/right scalar multiplication operation on tensors.
pub trait TensorScalarMulExt<E> {
    /// The type of the tensor representation.
    type A: TensorRepr;
    /// The type of the axis mapper.
    type M: AxisMapper;
    /// Creates a left scalar multiplication task.
    fn left_mul(self, lhs: E) -> Tensor<LeftScalarMulRepr<Self::A, E>, Self::M>;
    /// Creates a right scalar multiplication task.
    fn right_mul(self, rhs: E) -> Tensor<RightScalarMulRepr<Self::A, E>, Self::M>;
}

impl<T: ToTensorTuple<1>, E> TensorScalarMulExt<E> for T {
    type A = T::Repr;
    type M = T::Mapper;

    fn left_mul(self, lhs: E) -> Tensor<LeftScalarMulRepr<Self::A, E>, Self::M> {
        let (a, mapper) = self.to_tensor_tuple().into_raw();
        unsafe { Tensor::from_raw_unchecked(LeftScalarMulRepr { a, scalar: lhs }, mapper) }
    }
    fn right_mul(self, rhs: E) -> Tensor<RightScalarMulRepr<Self::A, E>, Self::M> {
        let (a, mapper) = self.to_tensor_tuple().into_raw();
        unsafe { Tensor::from_raw_unchecked(RightScalarMulRepr { a, scalar: rhs }, mapper) }
    }
}

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_scalar_mul {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Mul<(E,)> for $a
        where
            $a: ToTensorTuple<1>,
        {
            type Output = Tensor<RightScalarMulRepr<<Self as ToTensorTuple<1>>::Repr, E>, <Self as ToTensorTuple<1>>::Mapper>;
            fn mul(self, rhs: (E,)) -> Self::Output {
                self.to_tensor_tuple().right_mul(rhs.0)
            }
        }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Mul<$a> for (E,)
        where
            $a: ToTensorTuple<1>,
        {
            type Output = Tensor<LeftScalarMulRepr<<$a as ToTensorTuple<1>>::Repr, E>, <$a as ToTensorTuple<1>>::Mapper>;
            fn mul(self, rhs: $a) -> Self::Output {
                rhs.to_tensor_tuple().left_mul(self.0)
            }
        }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E:TensorScalar> Mul<E> for $a
        where
            $a: ToTensorTuple<1>,
        {
            type Output = Tensor<RightScalarMulRepr<<Self as ToTensorTuple<1>>::Repr, E>, <Self as ToTensorTuple<1>>::Mapper>;
            fn mul(self, rhs: E) -> Self::Output {
                self.to_tensor_tuple().right_mul(rhs)
            }
        }

    };
}
impl_scalar_mul!(Tensor<A, M>);
impl_scalar_mul!(&'a Tensor<A, M>,'a);
impl_scalar_mul!(&'a mut Tensor<A, M>,'a);

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_scalar_mul_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Mul<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>
            >>::Container;
            fn mul(self, rhs: (E,)) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let res = lhs_rt.ctx().execute(lhs * rhs);
                <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, lhs_rt)
                })
            }
        }

        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Mul<$a> for (E,)
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>
            >>::Container;
            fn mul(self, rhs: $a) -> Self::Output {
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();
                let res = rhs_rt.ctx().execute(self * rhs);
                <RT::Ctx as TensorContext<RT::Mk, 1, LeftScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, rhs_rt)
                })
            }
        }

        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E:TensorScalar> Mul<E> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>
            >>::Container;
            fn mul(self, rhs: E) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let res = lhs_rt.ctx().execute(lhs * rhs);
                <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarMulRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, lhs_rt)
                })
            }
        }
    };
}
impl_scalar_mul_runtime!(BoundTensor<A, M, RT>);
impl_scalar_mul_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_mul_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
