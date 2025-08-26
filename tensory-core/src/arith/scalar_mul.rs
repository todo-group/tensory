use core::ops::Mul;

use crate::tensor::{TensorBroker, Tensor, TensorRepr};

/// Raw context of left scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub trait LeftScalarMulContext<A: TensorRepr, E> {
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

pub struct TensorLeftScalarMul<M: TensorBroker, A: TensorRepr, E> {
    a: A,
    scalar: E,
    res_mgr: M,
}

impl<M: TensorBroker, A: TensorRepr, E> TensorLeftScalarMul<M, A, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: LeftScalarMulContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<M, C::Res>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.left_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mgr) })
    }
}

/// Raw context of right scalar multiplication operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub trait RightScalarMulContext<A: TensorRepr, E> {
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

pub struct TensorRightScalarMul<M: TensorBroker, A: TensorRepr, E> {
    a: A,
    scalar: E,
    res_mgr: M,
}

impl<M: TensorBroker, A: TensorRepr, E> TensorRightScalarMul<M, A, E> {
    // pub fn new(a: Tensor<LA, A>) -> Self {
    //     let (raw, legs) = a.into_raw();
    //     Self { a: raw, legs }
    // }
    pub fn with<C: RightScalarMulContext<A, E>>(
        self,
        context: C,
    ) -> Result<Tensor<M, C::Res>, C::Err> {
        let a = self.a;
        let scalar = self.scalar;

        let aconj = context.right_scalar_mul(a, scalar)?;

        Ok(unsafe { Tensor::from_raw_unchecked(aconj, self.res_mgr) })
    }
}

/// Raw context of scalar multiplication operation. (no left/right)
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub trait CommutativeScalarMulContext<A: TensorRepr, E> {
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
impl<A: TensorRepr, E, C: CommutativeScalarMulContext<A, E>> LeftScalarMulContext<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn left_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}
impl<A: TensorRepr, E, C: CommutativeScalarMulContext<A, E>> RightScalarMulContext<A, E> for C {
    type Res = C::Res;
    type Err = C::Err;

    fn right_scalar_mul(self, a: A, scalar: E) -> Result<Self::Res, Self::Err> {
        self.scalar_mul(a, scalar)
    }
}

impl<M: TensorBroker, T: TensorRepr> Tensor<M, T> {
    pub fn left_mul<E>(self, lhs: E) -> TensorLeftScalarMul<M, T, E> {
        let (a, mgr) = self.into_raw();
        TensorLeftScalarMul {
            a,
            scalar: lhs,
            res_mgr: mgr,
        }
    }
    pub fn right_mul<E>(self, lhs: E) -> TensorRightScalarMul<M, T, E> {
        let (a, mgr) = self.into_raw();
        TensorRightScalarMul {
            a,
            scalar: lhs,
            res_mgr: mgr,
        }
    }
}

trait TensorScalar {}

impl<M: TensorBroker, A: TensorRepr, E: TensorScalar> Mul<E> for Tensor<M, A> {
    type Output = TensorRightScalarMul<M, A, E>;

    fn mul(self, rhs: E) -> Self::Output {
        self.right_mul(rhs)
    }
}

// impl<M: AxisMgr, A: TensorRepr, E: TensorScalar> Mul<Tensor<M, A>> for E {
//     type Output = TensorLeftScalarMul<M, A, E>;

//     fn mul(self, rhs: Tensor<M, A>) -> Self::Output {
//         rhs.left_mul(self)
//     }
// }
