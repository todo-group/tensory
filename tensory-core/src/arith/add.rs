use core::ops::Add;

use crate::{
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    repr::TensorRepr,
    tensor::Tensor,
    tensor_with_runtime::{RuntimeError, TensorWithRuntime},
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
        axis_mapping: OverlayAxisMapping<2>,
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
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
impl<C: AddCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> AddCtx<Lhs, Rhs> for C {
    fn add(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let n_l = lhs.dim();
        let n_r = rhs.dim();
        let n = axis_mapping.dim();
        if n_l != n || n_r != n {
            panic!("axis_origin must match the number of axes with lhs and rhs");
        }
        unsafe { self.add_unchecked(lhs, rhs, axis_mapping) }
    }
}

pub struct TensorAdd<L: TensorRepr, R: TensorRepr, B: AxisMapper> {
    lhs: L,
    rhs: R,
    axis_origin: OverlayAxisMapping<2>,
    res_mgr: B,
}
impl<L: TensorRepr, R: TensorRepr, B: AxisMapper> TensorAdd<L, R, B> {
    pub unsafe fn from_raw_unchecked(
        lhs: L,
        rhs: R,
        axis_origin: OverlayAxisMapping<2>,
        res_mgr: B,
    ) -> Self {
        Self {
            lhs,
            rhs,
            axis_origin,
            res_mgr,
        }
    }
    pub fn from_raw(lhs: L, rhs: R, axis_origin: OverlayAxisMapping<2>, res_broker: B) -> Self {
        let n_l = lhs.dim();
        let n_r = rhs.dim();
        let n = axis_origin.dim();
        let n_m = res_broker.dim();
        if n_l != n || n_r != n || n_m != n {
            panic!("lhs, rhs, axis_origin, res_broker must match the number of axes");
        }
        unsafe { Self::from_raw_unchecked(lhs, rhs, axis_origin, res_broker) }
    }

    pub fn with<C: AddCtx<L, R>>(self, context: C) -> Result<Tensor<C::Res, B>, C::Err> {
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.add_unchecked(self.lhs, self.rhs, self.axis_origin)?,
                self.res_mgr,
            )
        })
    }
}

impl<L: TensorRepr, R: TensorRepr, B: OverlayMapper<2>> Add<Tensor<R, B>> for Tensor<L, B> {
    type Output = Result<TensorAdd<L, R, B>, B::Err>;

    fn add(self, rhs: Tensor<R, B>) -> Self::Output {
        let (lhs, lhs_legs) = self.into_raw();
        let (rhs, rhs_legs) = rhs.into_raw();

        let (res_mgr, axis_origin) = B::overlay([lhs_legs, rhs_legs])?;

        Ok(TensorAdd {
            lhs,
            rhs,
            axis_origin,
            res_mgr,
        })
    }
}

impl<'rt, L: TensorRepr, R: TensorRepr, B: OverlayMapper<2>, RT: Eq>
    Add<TensorWithRuntime<'rt, R, B, RT>> for TensorWithRuntime<'rt, L, B, RT>
where
    &'rt RT: AddCtxImpl<L, R>,
{
    type Output = Result<
        TensorWithRuntime<'rt, <&'rt RT as AddCtxImpl<L, R>>::Res, B, RT>,
        RuntimeError<B::Err, <&'rt RT as AddCtxImpl<L, R>>::Err>,
    >;

    fn add(self, rhs: TensorWithRuntime<'rt, R, B, RT>) -> Self::Output {
        let (lhs, lhs_rt) = self.into_raw();
        let (rhs, rhs_rt) = rhs.into_raw();

        if lhs_rt != rhs_rt {
            return Err(RuntimeError::Runtime);
        }
        let res = (lhs + rhs)
            .map_err(|e| RuntimeError::Axis(e))?
            .with(lhs_rt)
            .map_err(|e| RuntimeError::Ctx(e))?;
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
