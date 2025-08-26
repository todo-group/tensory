use core::ops::Add;

use crate::tensor::{OverlayAxisOrigin, OverlayBroker, Tensor, TensorBroker, TensorRepr};

/// Raw context of addition operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor has same number of axes as the input tensors.
pub unsafe trait AddCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis pairs, and returns the result tensor each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that `axis_origin` is for the number of axes same as the input tensors.
    ///
    /// the implementor must ensure the result tensor has same number of axes as the input tensors.
    unsafe fn add_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: OverlayAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err>;
}
/// Safe version if AddCtxImpl.
///
/// The blanket implementation checks both input and output.
pub trait AddCtx<Lhs: TensorRepr, Rhs: TensorRepr>: AddCtxImpl<Lhs, Rhs> {
    fn add(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: OverlayAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err>;
}
impl<C: AddCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> AddCtx<Lhs, Rhs> for C {
    fn add(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: OverlayAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        let n_l = lhs.dim();
        let n_r = rhs.dim();
        let n = axis_origin.len();
        if n_l != n || n_r != n {
            panic!("axis_origin must match the number of axes with lhs and rhs");
        }
        unsafe { self.add_unchecked(lhs, rhs, axis_origin) }
    }
}

pub struct TensorAdd<M: TensorBroker, L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    axis_origin: OverlayAxisOrigin<2>,
    res_mgr: M,
}
impl<M: TensorBroker, L: TensorRepr, R: TensorRepr> TensorAdd<M, L, R> {
    // unsafe fn from_raw_unchecked(
    //     lhs: L,
    //     rhs: R,
    //     axis_origin: OverlayAxisOrigin<2>,
    //     res_mgr: M,
    // ) -> Self {
    //     Self {
    //         lhs,
    //         rhs,
    //         axis_origin,
    //         res_mgr,
    //     }
    // }

    pub fn with<C: AddCtx<L, R>>(self, context: C) -> Result<Tensor<M, C::Res>, C::Err> {
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.add_unchecked(self.lhs, self.rhs, self.axis_origin)?,
                self.res_mgr,
            )
        })
    }
}

impl<M: OverlayBroker<2>, L: TensorRepr, R: TensorRepr> Add<Tensor<M, R>> for Tensor<M, L> {
    type Output = Result<TensorAdd<M, L, R>, M::Err>;

    fn add(self, rhs: Tensor<M, R>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_mgr, axis_origin) = M::overlay([lhs_legs, rhs_legs])?;

        Ok(TensorAdd {
            lhs,
            rhs,
            axis_origin,
            res_mgr,
        })
    }
}
