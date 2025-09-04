use core::ops::Add;

use thiserror::Error;

use crate::tensor::{
    OverlayAxisOrigin, OverlayBroker, Tensor, TensorBroker, TensorRepr,
    tensor_with_runtime::TensorWithRuntime,
};

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
    pub unsafe fn from_raw_unchecked(
        lhs: L,
        rhs: R,
        axis_origin: OverlayAxisOrigin<2>,
        res_mgr: M,
    ) -> Self {
        Self {
            lhs,
            rhs,
            axis_origin,
            res_mgr,
        }
    }
    pub fn from_raw(lhs: L, rhs: R, axis_origin: OverlayAxisOrigin<2>, res_mgr: M) -> Self {
        let n_l = lhs.dim();
        let n_r = rhs.dim();
        let n = axis_origin.len();
        let n_m = res_mgr.len();
        if n_l != n || n_r != n || n_m != n {
            panic!("lhs, rhs, axis_origin, res_mgr must match the number of axes");
        }
        unsafe { Self::from_raw_unchecked(lhs, rhs, axis_origin, res_mgr) }
    }

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

#[derive(Error, Debug)]
pub enum RuntimeAddError<AE, CE> {
    #[error("Runtime error")]
    Runtime,
    #[error("Axis error: {0}")]
    Axis(AE),
    #[error("Context error: {0}")]
    Ctx(CE),
}

impl<'rt, B: OverlayBroker<2>, L: TensorRepr, R: TensorRepr, RT: Eq>
    Add<TensorWithRuntime<'rt, B, R, RT>> for TensorWithRuntime<'rt, B, L, RT>
where
    &'rt RT: AddCtxImpl<L, R>,
{
    type Output = Result<
        TensorWithRuntime<'rt, B, <&'rt RT as AddCtxImpl<L, R>>::Res, RT>,
        RuntimeAddError<B::Err, <&'rt RT as AddCtxImpl<L, R>>::Err>,
    >;

    fn add(self, rhs: TensorWithRuntime<'rt, B, R, RT>) -> Self::Output {
        let (lhs, lhs_rt) = self.into_raw();
        let (rhs, rhs_rt) = rhs.into_raw();

        if lhs_rt != rhs_rt {
            return Err(RuntimeAddError::Runtime);
        }
        let res = (lhs + rhs)
            .map_err(|e| RuntimeAddError::Axis(e))?
            .with(lhs_rt)
            .map_err(|e| RuntimeAddError::Ctx(e))?;
        Ok(TensorWithRuntime::from_raw(res, lhs_rt))
    }
}

// impl<M: OverlayBroker<2>, L: TensorRepr, R: TensorRepr> Add<&Tensor<M, R>> for Tensor<M, L> {
//     type Output = Result<TensorAdd<M, L, R>, M::Err>;

//     fn add(self, rhs: &Tensor<M, R>) -> Self::Output {
//         let (lhs, lhs_legs) = self.into_raw();
//         let (rhs, rhs_legs) = rhs.view().into_raw();

//         let (res_mgr, axis_origin) = M::overlay([lhs_legs, rhs_legs])?;

//         Ok(TensorAdd {
//             lhs,
//             rhs,
//             axis_origin,
//             res_mgr,
//         })
//     }
// }
