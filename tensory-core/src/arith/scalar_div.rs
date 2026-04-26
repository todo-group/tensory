use core::ops::Div;

use crate::{
    arith::TensorScalar,
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::AxisMapper,
    repr::{ReprContext, TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

/// Intermediate task representation for left scalar division (scalar \ tensor) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LeftScalarDivRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> LeftScalarDivRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for LeftScalarDivRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for LeftScalarDivRepr<A, E> {}

/// Intermediate task representation for right scalar division (tensor / scalar) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RightScalarDivRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> RightScalarDivRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for RightScalarDivRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for RightScalarDivRepr<A, E> {}

/// Commutative task representation for commutative scalar division (tensor / scalar) operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CommutativeScalarDivRepr<A: TensorRepr, E> {
    a: A,
    scalar: E,
}

impl<A: TensorRepr, E> CommutativeScalarDivRepr<A, E> {
    pub fn from_raw(a: A, scalar: E) -> Self {
        Self { a, scalar }
    }
    pub fn into_raw(self) -> (A, E) {
        (self.a, self.scalar)
    }
}

unsafe impl<A: TensorRepr, E> TensorTupleRepr<1> for CommutativeScalarDivRepr<A, E> {
    fn naxeses(&self) -> [usize; 1] {
        [self.a.naxes()]
    }
}
impl<A: TensorRepr, E> IsTask for CommutativeScalarDivRepr<A, E> {}

impl<A: TensorRepr, E, Mk, C: Context<Mk, CommutativeScalarDivRepr<A, E>>>
    Context<Mk, LeftScalarDivRepr<A, E>> for C
{
    type Output = <C as Context<Mk, CommutativeScalarDivRepr<A, E>>>::Output;

    fn execute(self, task: LeftScalarDivRepr<A, E>) -> Self::Output {
        self.execute(CommutativeScalarDivRepr {
            a: task.a,
            scalar: task.scalar,
        })
    }
}
unsafe impl<A: TensorRepr, E, Mk, C: ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>
    ReprContext<Mk, 1, LeftScalarDivRepr<A, E>> for C
{
    type Repr = <C as ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>::Repr;
    type CType = <C as ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>::CType;
}

impl<A: TensorRepr, E, Mk, C: Context<Mk, CommutativeScalarDivRepr<A, E>>>
    Context<Mk, RightScalarDivRepr<A, E>> for C
{
    type Output = <C as Context<Mk, CommutativeScalarDivRepr<A, E>>>::Output;

    fn execute(self, task: RightScalarDivRepr<A, E>) -> Self::Output {
        self.execute(CommutativeScalarDivRepr {
            a: task.a,
            scalar: task.scalar,
        })
    }
}
unsafe impl<A: TensorRepr, E, Mk, C: ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>
    ReprContext<Mk, 1, RightScalarDivRepr<A, E>> for C
{
    type Repr = <C as ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>::Repr;
    type CType = <C as ReprContext<Mk, 1, CommutativeScalarDivRepr<A, E>>>::CType;
}

/// Extension trait for left/right scalar division operation on tensors.
pub trait TensorScalarDivExt<E> {
    /// The type of the tensor representation.
    type A: TensorRepr;
    /// The type of the axis mapper.
    type M: AxisMapper;
    /// Creates a left scalar division task.
    fn left_div(self, lhs: E) -> Tensor<LeftScalarDivRepr<Self::A, E>, Self::M>;
    /// Creates a right scalar division task.
    fn right_div(self, rhs: E) -> Tensor<RightScalarDivRepr<Self::A, E>, Self::M>;
}

impl<T: ToTensorTuple<1>, E> TensorScalarDivExt<E> for T {
    type A = T::Repr;
    type M = T::Mapper;

    fn left_div(self, lhs: E) -> Tensor<LeftScalarDivRepr<Self::A, E>, Self::M> {
        let (a, mapper) = self.to_tensor_tuple().into_raw();
        unsafe { Tensor::from_raw_unchecked(LeftScalarDivRepr { a, scalar: lhs }, mapper) }
    }
    fn right_div(self, rhs: E) -> Tensor<RightScalarDivRepr<Self::A, E>, Self::M> {
        let (a, mapper) = self.to_tensor_tuple().into_raw();
        unsafe { Tensor::from_raw_unchecked(RightScalarDivRepr { a, scalar: rhs }, mapper) }
    }
}

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_scalar_div {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E> Div<(E,)> for $a
        where
            $a: ToTensorTuple<1>,
        {
            type Output = Tensor<RightScalarDivRepr<<Self as ToTensorTuple<1>>::Repr, E>, <Self as ToTensorTuple<1>>::Mapper>;
            fn div(self, rhs: (E,)) -> Self::Output {
                self.to_tensor_tuple().right_div(rhs.0)
            }
        }
        impl<$($life,)* A: TensorRepr, M: AxisMapper,E:TensorScalar> Div<E> for $a
        where
            $a: ToTensorTuple<1>,
        {
            type Output = Tensor<RightScalarDivRepr<<Self as ToTensorTuple<1>>::Repr, E>, <Self as ToTensorTuple<1>>::Mapper>;
            fn div(self, rhs: E) -> Self::Output {
                self.to_tensor_tuple().right_div(rhs)
            }
        }

    };
}
impl_scalar_div!(Tensor<A, M>);
impl_scalar_div!(&'a Tensor<A, M>,'a);
impl_scalar_div!(&'a mut Tensor<A, M>,'a);

// 3 combinations of A being owned/view/view_mut
macro_rules! impl_scalar_div_runtime {
    ($a:ty $(,$life:lifetime)*) => {
        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E> Div<(E,)> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>
            >>::Container;
            fn div(self, rhs: (E,)) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let res = lhs_rt.ctx().execute(lhs / rhs);
                <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, lhs_rt)
                })
            }
        }

        impl<$($life,)* A: TensorRepr, M: AxisMapper, RT:Runtime, E:TensorScalar> Div<E> for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>,
            >,
        {
            type Output = <<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType as ContainerImpl<
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::Repr,M,RT>
            >>::Container;
            fn div(self, rhs: E) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let res = lhs_rt.ctx().execute(lhs / rhs);
                <RT::Ctx as TensorContext<RT::Mk, 1, RightScalarDivRepr<<$a as ToBoundTensorTuple<1>>::Repr, E>, M>>::CType::map(res, |res| {
                    BoundTensorTuple::from_raw(res, lhs_rt)
                })
            }
        }
    };
}
impl_scalar_div_runtime!(BoundTensor<A, M, RT>);
impl_scalar_div_runtime!(&'a BoundTensor<A, M, RT>,'a);
impl_scalar_div_runtime!(&'a mut BoundTensor<A, M, RT>,'a);
