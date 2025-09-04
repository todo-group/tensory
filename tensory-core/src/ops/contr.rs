use core::ops::Mul;

use crate::tensor::{
    ConnectAxisOrigin, ConnectBroker, Tensor, TensorBroker, TensorRepr, ViewableRepr,
};

/// Raw context of contraction operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `ContractionAxisProvenance` is valid for the given tensors.
pub unsafe trait MulCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs contraction operation on the tensors `lhs` and `rhs` with the given axis pairs, and returns the result tensor and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axis pairs are valid for the given tensors.
    ///
    /// the implementor must ensure the list of `ContractionAxisProvenance` is valid for the given tensors.
    unsafe fn mul_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err>;
}
pub enum MulAxisOrigin {
    Lhs(usize),
    Rhs(usize),
}

/// Safe version if ContractionContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait MulCtx<Lhs: TensorRepr, Rhs: TensorRepr>: MulCtxImpl<Lhs, Rhs> {
    fn mul(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err>;
}
impl<C: MulCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> MulCtx<Lhs, Rhs> for C {
    fn mul(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        if axis_origin.in_lens() != [lhs.dim(), rhs.dim()] {
            panic!("axis_origin must match the number of axes with lhs and rhs");
        }

        unsafe { self.mul_unchecked(lhs, rhs, axis_origin) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMul<M: TensorBroker, L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    res_mgr: M,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<B: TensorBroker, L: TensorRepr, R: TensorRepr> TensorMul<B, L, R> {
    // pub fn from_raw(lhs: Tensor<B, L>, rhs: Tensor<B, R>) -> Self where B:{
    //     let (lhs, lhs_legs) = lhs.into_raw();
    //     let (rhs, rhs_legs) = rhs.into_raw();

    //     let (intermediate, axis_pairs) = B::(lhs_legs, rhs_legs);

    //     Self {
    //         lhs,
    //         rhs,
    //         intermediate,
    //         axis_pairs,
    //     }
    // }
    pub fn with<C: MulCtx<L, R>>(self, context: C) -> Result<Tensor<B, C::Res>, C::Err> {
        //println!("lhs: {:?}, rhs: {:?}", lhs_legs, rhs_legs);
        //println!("idx_pairs: {:?}", idx_pairs);

        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.mul_unchecked(self.lhs, self.rhs, self.axis_origin)?,
                self.res_mgr,
            )
        })
    }
}

impl<M: ConnectBroker<2>, L: TensorRepr, R: TensorRepr> Mul<Tensor<M, R>> for Tensor<M, L> {
    type Output = Result<TensorMul<M, L, R>, M::Err>;

    fn mul(self, rhs: Tensor<M, R>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_mgr, axis_origin) = M::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_mgr,
            axis_origin,
        })
    }
}

impl<'a, M: ConnectBroker<2> + Clone, L: TensorRepr, R: TensorRepr + ViewableRepr<'a>>
    Mul<&'a Tensor<M, R>> for Tensor<M, L>
{
    type Output = Result<TensorMul<M, L, R::View>, M::Err>;

    fn mul(self, rhs: &'a Tensor<M, R>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.view().into_raw();

        let (res_mgr, axis_origin) = M::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_mgr,
            axis_origin,
        })
    }
}

impl<
    'a,
    M: ConnectBroker<2> + Clone,
    L: TensorRepr + ViewableRepr<'a>,
    R: TensorRepr + ViewableRepr<'a>,
> Mul<&'a Tensor<M, R>> for &'a Tensor<M, L>
{
    type Output = Result<TensorMul<M, L::View, R::View>, M::Err>;

    fn mul(self, rhs: &'a Tensor<M, R>) -> Self::Output {
        let (lhs, lhs_legs) = self.view().into_raw();
        let (rhs, rhs_legs) = rhs.view().into_raw();

        let (res_mgr, axis_origin) = M::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_mgr,
            axis_origin,
        })
    }
}

impl<'a, M: ConnectBroker<2> + Clone, L: TensorRepr + ViewableRepr<'a>, R: TensorRepr>
    Mul<Tensor<M, R>> for &'a Tensor<M, L>
{
    type Output = Result<TensorMul<M, L::View, R>, M::Err>;

    fn mul(self, rhs: Tensor<M, R>) -> Self::Output {
        let (lhs, lhs_legs) = self.view().into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_mgr, axis_origin) = M::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_mgr,
            axis_origin,
        })
    }
}
