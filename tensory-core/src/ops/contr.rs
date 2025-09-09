use core::ops::Mul;

use crate::tensor::{
    AsViewMutRepr, AsViewRepr, ConnectAxisOrigin, ConnectBroker, RuntimeError, Tensor,
    TensorBroker, TensorRepr, TensorWithRuntime,
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
pub struct TensorMul<L: TensorRepr, R: TensorRepr, B: TensorBroker> {
    lhs: L,
    rhs: R,
    res_broker: B,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorRepr, R: TensorRepr, B: TensorBroker> TensorMul<L, R, B> {
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
    pub fn with<C: MulCtx<L, R>>(self, context: C) -> Result<Tensor<C::Res, B>, C::Err> {
        //println!("lhs: {:?}, rhs: {:?}", lhs_legs, rhs_legs);
        //println!("idx_pairs: {:?}", idx_pairs);

        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.mul_unchecked(self.lhs, self.rhs, self.axis_origin)?,
                self.res_broker,
            )
        })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

impl<L: TensorRepr, R: TensorRepr, B: ConnectBroker<2>> Mul<Tensor<R, B>> for Tensor<L, B> {
    type Output = Result<TensorMul<L, R, B>, B::Err>;

    fn mul(self, rhs: Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<'r, L: TensorRepr, R: TensorRepr + AsViewRepr<'r>, B: ConnectBroker<2> + Clone>
    Mul<&'r Tensor<R, B>> for Tensor<L, B>
{
    type Output = Result<TensorMul<L, R::View, B>, B::Err>;

    fn mul(self, rhs: &'r Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.view().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<'l, L: TensorRepr + AsViewRepr<'l>, R: TensorRepr, B: ConnectBroker<2> + Clone>
    Mul<Tensor<R, B>> for &'l Tensor<L, B>
{
    type Output = Result<TensorMul<L::View, R, B>, B::Err>;

    fn mul(self, rhs: Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view().into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<
    'l,
    'r,
    L: TensorRepr + AsViewRepr<'l>,
    R: TensorRepr + AsViewRepr<'r>,
    B: ConnectBroker<2> + Clone,
> Mul<&'r Tensor<R, B>> for &'l Tensor<L, B>
{
    type Output = Result<TensorMul<L::View, R::View, B>, B::Err>;

    fn mul(self, rhs: &'r Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view().into_raw();
        let (rhs, rhs_legs) = rhs.view().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<'r, L: TensorRepr, R: TensorRepr + AsViewMutRepr<'r>, B: ConnectBroker<2> + Clone>
    Mul<&'r mut Tensor<R, B>> for Tensor<L, B>
{
    type Output = Result<TensorMul<L, R::ViewMut, B>, B::Err>;

    fn mul(self, rhs: &'r mut Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.view_mut().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<'l, L: TensorRepr + AsViewMutRepr<'l>, R: TensorRepr, B: ConnectBroker<2> + Clone>
    Mul<Tensor<R, B>> for &'l mut Tensor<L, B>
{
    type Output = Result<TensorMul<L::ViewMut, R, B>, B::Err>;

    fn mul(self, rhs: Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view_mut().into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<
    'l,
    'r,
    L: TensorRepr + AsViewMutRepr<'l>,
    R: TensorRepr + AsViewRepr<'r>,
    B: ConnectBroker<2> + Clone,
> Mul<&'r Tensor<R, B>> for &'l mut Tensor<L, B>
{
    type Output = Result<TensorMul<L::ViewMut, R::View, B>, B::Err>;

    fn mul(self, rhs: &'r Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view_mut().into_raw();
        let (rhs, rhs_legs) = rhs.view().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<
    'l,
    'r,
    L: TensorRepr + AsViewRepr<'l>,
    R: TensorRepr + AsViewMutRepr<'r>,
    B: ConnectBroker<2> + Clone,
> Mul<&'r mut Tensor<R, B>> for &'l Tensor<L, B>
{
    type Output = Result<TensorMul<L::View, R::ViewMut, B>, B::Err>;

    fn mul(self, rhs: &'r mut Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view().into_raw();
        let (rhs, rhs_legs) = rhs.view_mut().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}
impl<
    'l,
    'r,
    L: TensorRepr + AsViewMutRepr<'l>,
    R: TensorRepr + AsViewMutRepr<'r>,
    B: ConnectBroker<2> + Clone,
> Mul<&'r mut Tensor<R, B>> for &'l mut Tensor<L, B>
{
    type Output = Result<TensorMul<L::ViewMut, R::ViewMut, B>, B::Err>;

    fn mul(self, rhs: &'r mut Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.view_mut().into_raw();
        let (rhs, rhs_legs) = rhs.view_mut().into_raw();

        let (res_broker, axis_origin) = B::connect([lhs_legs, rhs_legs])?;

        Ok(TensorMul {
            lhs,
            rhs,
            res_broker,
            axis_origin,
        })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

impl<'rt, L: TensorRepr, R: TensorRepr, B: ConnectBroker<2>, RT: Eq>
    Mul<TensorWithRuntime<'rt, R, B, RT>> for TensorWithRuntime<'rt, L, B, RT>
where
    &'rt RT: MulCtxImpl<L, R>,
{
    type Output = Result<
        TensorWithRuntime<'rt, <&'rt RT as MulCtxImpl<L, R>>::Res, B, RT>,
        RuntimeError<B::Err, <&'rt RT as MulCtxImpl<L, R>>::Err>,
    >;

    fn mul(self, rhs: TensorWithRuntime<'rt, R, B, RT>) -> Self::Output {
        let (lhs, lhs_rt) = self.into_raw();
        let (rhs, rhs_rt) = rhs.into_raw();

        if lhs_rt != rhs_rt {
            return Err(RuntimeError::Runtime);
        }
        let res = (lhs * rhs)
            .map_err(|e| RuntimeError::Axis(e))?
            .with(lhs_rt)
            .map_err(|e| RuntimeError::Ctx(e))?;
        Ok(TensorWithRuntime::from_raw(res, lhs_rt))
    }
}
impl<'l, 'r, 'rt, B: ConnectBroker<2> + Clone, L: AsViewRepr<'l>, R: AsViewRepr<'r>, RT: Eq>
    Mul<&'r TensorWithRuntime<'rt, R, B, RT>> for &'l TensorWithRuntime<'rt, L, B, RT>
where
    &'rt RT: MulCtxImpl<L::View, R::View>,
{
    type Output = Result<
        TensorWithRuntime<'rt, <&'rt RT as MulCtxImpl<L::View, R::View>>::Res, B, RT>,
        RuntimeError<B::Err, <&'rt RT as MulCtxImpl<L::View, R::View>>::Err>,
    >;

    fn mul(self, rhs: &'r TensorWithRuntime<'rt, R, B, RT>) -> Self::Output {
        let lhs_rt = self.runtime();
        let rhs_rt = rhs.runtime();
        let lhs = self.tensor().view();
        let rhs = rhs.tensor().view();

        if lhs_rt != rhs_rt {
            return Err(RuntimeError::Runtime);
        }

        let res = (lhs * rhs)
            .map_err(|e| RuntimeError::Axis(e))?
            .with(lhs_rt)
            .map_err(|e| RuntimeError::Ctx(e))?;
        Ok(TensorWithRuntime::from_raw(res, lhs_rt))
    }
}
