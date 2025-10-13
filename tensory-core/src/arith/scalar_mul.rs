use core::ops::Mul;

use crate::{mapper::AxisMapper, repr::TensorRepr, tensor::Tensor};

/// Raw context of left scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait LeftScalarMulContext<A: TensorRepr, E> {
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

pub struct TensorLeftScalarMul<A: TensorRepr, B: AxisMapper, E> {
    a: A,
    scalar: E,
    res_broker: B,
}

impl<A: TensorRepr, B: AxisMapper, E> TensorLeftScalarMul<A, B, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: LeftScalarMulContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<C::Res, B>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.left_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of right scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait RightScalarMulContext<A: TensorRepr, E> {
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

impl<A: TensorRepr, B: AxisMapper, E> TensorRightScalarMul<A, B, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: RightScalarMulContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<C::Res, B>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.right_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_broker) })
    }
}

/// Raw context of scalar multiplication operation. (no left/right)
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait CommutativeScalarMulContext<A: TensorRepr, E> {
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
unsafe impl<A: TensorRepr, E, C: CommutativeScalarMulContext<A, E>> LeftScalarMulContext<A, E>
    for C
{
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}
unsafe impl<A: TensorRepr, E, C: CommutativeScalarMulContext<A, E>> RightScalarMulContext<A, E>
    for C
{
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}

impl<A: TensorRepr, B: AxisMapper> Tensor<A, B> {
    pub fn left_mul<E>(self, lhs: E) -> TensorLeftScalarMul<A, B, E> {
        let (a, broker) = self.into_raw();
        TensorLeftScalarMul {
            a,
            scalar: lhs,
            res_broker: broker,
        }
    }
    pub fn right_mul<E>(self, rhs: E) -> TensorRightScalarMul<A, B, E> {
        let (a, broker) = self.into_raw();
        TensorRightScalarMul {
            a,
            scalar: rhs,
            res_broker: broker,
        }
    }
}

impl<A: TensorRepr, B: AxisMapper, E> Mul<(E,)> for Tensor<A, B> {
    type Output = TensorRightScalarMul<A, B, E>;

    fn mul(self, rhs: (E,)) -> Self::Output {
        self.right_mul(rhs.0)
    }
}

impl<A: TensorRepr, B: AxisMapper, E> Mul<Tensor<A, B>> for (E,) {
    type Output = TensorLeftScalarMul<A, B, E>;

    fn mul(self, rhs: Tensor<A, B>) -> Self::Output {
        rhs.left_mul(self.0)
    }
}
