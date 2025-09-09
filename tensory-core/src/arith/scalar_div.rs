use core::ops::Div;

use crate::tensor::{Tensor, TensorBroker, TensorRepr};

/// Raw context of left scalar division operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait LeftScalarDivContext<A: TensorRepr, E> {
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

pub struct TensorLeftScalarDiv<A: TensorRepr, B: TensorBroker, E> {
    a: A,
    scalar: E,
    res_broker: B,
}

impl<A: TensorRepr, B: TensorBroker, E> TensorLeftScalarDiv<A, B, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: LeftScalarDivContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<C::Res, B>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.left_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of right scalar division operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait RightScalarDivContext<A: TensorRepr, E> {
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

pub struct TensorRightScalarDiv<A: TensorRepr, B: TensorBroker, E> {
    a: A,
    scalar: E,
    res_broker: B,
}

impl<A: TensorRepr, B: TensorBroker, E> TensorRightScalarDiv<A, B, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: RightScalarDivContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<C::Res, B>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.right_scalar_div(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of scalar division operation. (no left/right)
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait CommutativeScalarDivContext<A: TensorRepr, E> {
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
unsafe impl<A: TensorRepr, E, C: CommutativeScalarDivContext<A, E>> LeftScalarDivContext<A, E>
    for C
{
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarDivContext<A, E>> RightScalarDivContext<A, E>
    for C
{
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_div(a, scalar)
    }
}

impl<A: TensorRepr, B: TensorBroker> Tensor<A, B> {
    pub fn left_div<E>(self, lhs: E) -> TensorLeftScalarDiv<A, B, E> {
        let (a, broker) = self.into_raw();
        TensorLeftScalarDiv {
            a,
            scalar: lhs,
            res_broker: broker,
        }
    }
    pub fn right_div<E>(self, rhs: E) -> TensorRightScalarDiv<A, B, E> {
        let (a, broker) = self.into_raw();
        TensorRightScalarDiv {
            a,
            scalar: rhs,
            res_broker: broker,
        }
    }
}

// trait TensorScalar {}
// impl TensorScalar for bool {}
// impl TensorScalar for i8 {}
// impl TensorScalar for i16 {}
// impl TensorScalar for i32 {}
// impl TensorScalar for i64 {}
// impl TensorScalar for i128 {}
// impl TensorScalar for isize {}
// impl TensorScalar for u8 {}
// impl TensorScalar for u16 {}
// impl TensorScalar for u32 {}
// impl TensorScalar for u64 {}
// impl TensorScalar for u128 {}
// impl TensorScalar for usize {}
// //impl TensorScalar for f16 {}
// impl TensorScalar for f32 {}
// impl TensorScalar for f64 {}
// //impl TensorScalar for f128 {}

impl<A: TensorRepr, B: TensorBroker, E> Div<(E,)> for Tensor<A, B> {
    type Output = TensorRightScalarDiv<A, B, E>;

    fn div(self, rhs: (E,)) -> Self::Output {
        self.right_div(rhs.0)
    }
}

// unfortunately we not have left_div common format
