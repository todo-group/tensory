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
        let n_l = lhs.naxes();
        let n_r = rhs.naxes();
        let n = axis_mapping.dim();
        if n_l != n || n_r != n {
            panic!("axis_mapping must match the number of axes with lhs and rhs");
        }
        unsafe { self.add_unchecked(lhs, rhs, axis_mapping) }
    }
}

pub struct TensorAdd<L: TensorRepr, R: TensorRepr, M: AxisMapper> {
    lhs: L,
    rhs: R,
    res_mapper: M,
    axis_mapping: OverlayAxisMapping<2>,
}
impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TensorAdd<L, R, M> {
    pub fn by_manager(
        lhs: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> (M, OverlayAxisMapping<2>),
    ) -> Self {
        let (lhs, lhs_mapper) = lhs.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_mapping) = manager(lhs_mapper, rhs_mapper);

        Self {
            lhs,
            rhs,
            res_mapper,
            axis_mapping,
        }
    }
    pub fn try_by_manager<E>(
        lhs: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> Result<(M, OverlayAxisMapping<2>), E>,
    ) -> Result<Self, E> {
        let (lhs, lhs_mapper) = lhs.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_origin) = manager(lhs_mapper, rhs_mapper)?;

        Ok(Self {
            lhs,
            rhs,
            res_mapper,
            axis_mapping: axis_origin,
        })
    }

    pub fn with<C: AddCtx<L, R>>(self, context: C) -> Result<Tensor<C::Res, M>, C::Err> {
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.add_unchecked(self.lhs, self.rhs, self.axis_mapping)?,
                self.res_mapper,
            )
        })
    }
}

// impl<L: TensorRepr, R: TensorRepr, B: OverlayMapper<2>> Add<Tensor<R, B>> for Tensor<L, B> {
//     type Output = Result<TensorAdd<L, R, B>, B::Err>;

//     fn add(self, rhs: Tensor<R, B>) -> Self::Output {
//         let (lhs, lhs_legs) = self.into_raw();
//         let (rhs, rhs_legs) = rhs.into_raw();

//         let (res_mgr, axis_origin) = B::overlay([lhs_legs, rhs_legs])?;

//         Ok(TensorAdd {
//             lhs,
//             rhs,
//             axis_origin,
//             res_mgr,
//         })
//     }
// }

// impl<L: TensorRepr, R: TensorRepr, B: OverlayMapper<2>, RT: Eq> Add<TensorWithRuntime<R, B, RT>>
//     for TensorWithRuntime<L, B, RT>
// where
//     RT: AddCtxImpl<L, R>,
// {
//     type Output = Result<
//         TensorWithRuntime<<RT as AddCtxImpl<L, R>>::Res, B, RT>,
//         RuntimeError<B::Err, <&'rt RT as AddCtxImpl<L, R>>::Err>,
//     >;

//     fn add(self, rhs: TensorWithRuntime<'rt, R, B, RT>) -> Self::Output {
//         let (lhs, lhs_rt) = self.into_raw();
//         let (rhs, rhs_rt) = rhs.into_raw();

//         if lhs_rt != rhs_rt {
//             return Err(RuntimeError::Runtime);
//         }
//         let res = (lhs + rhs)
//             .map_err(|e| RuntimeError::Axis(e))?
//             .with(lhs_rt)
//             .map_err(|e| RuntimeError::Ctx(e))?;
//         Ok(TensorWithRuntime::from_raw(res, lhs_rt))
//     }
// }

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

// 9 combinations of Lhs/Rhs being owned/view/view_mut

use crate::tensor::ToTensor;

macro_rules! impl_mul {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>> Add<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                TensorAdd<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr, M>,
                <M as OverlayMapper<2>>::Err,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let lhs = ToTensor::to_tensor(self);
                let rhs = ToTensor::to_tensor(rhs);
                TensorAdd::try_by_manager(lhs, rhs, |l, r| OverlayMapper::<2>::overlay([l, r]))
            }
        }
    };
}

impl_mul!(Tensor<L, M>, Tensor<R, M>);
impl_mul!(&'l Tensor<L, M>, Tensor<R, M>,'l);
impl_mul!(&'l mut Tensor<L, M>, Tensor<R, M>,'l);
impl_mul!(Tensor<L, M>, &'r Tensor<R, M>,'r);
impl_mul!(&'l Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_mul!(&'l mut Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_mul!(Tensor<L, M>, &'r mut Tensor<R, M>,'r);
impl_mul!(&'l Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);
impl_mul!(&'l mut Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);

pub trait AddRuntime<Lhs: TensorRepr, Rhs: TensorRepr> {
    type Ctx: AddCtxImpl<Lhs, Rhs>;
    fn add_ctx(self) -> Self::Ctx;
}

// // 9 combinations of Lhs/Rhs being owned/view/view_mut
use crate::tensor_with_runtime::ToTensorWithRuntime;

macro_rules! impl_mul_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT> Add<$r> for $l
        where
            $l: ToTensorWithRuntime<Mapper = M, Runtime = RT>,
            $r: ToTensorWithRuntime<Mapper = M, Runtime = RT>,
            RT: Copy + Eq + AddRuntime<<$l as ToTensorWithRuntime>::Repr, <$r as ToTensorWithRuntime>::Repr>,
        {
            type Output = Result<
                TensorWithRuntime<
                    <<RT as AddRuntime<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Ctx as AddCtxImpl<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    <M as OverlayMapper<2>>::Err,
                    <<RT as AddRuntime<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Ctx as AddCtxImpl<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Err,
                >,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_tensor_with_runtime().into_raw();
                let (rhs, rhs_rt) = rhs.to_tensor_with_runtime().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeError::Runtime);
                }
                let res = (lhs + rhs)
                    .map_err(RuntimeError::Axis)?
                    .with(lhs_rt.add_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(TensorWithRuntime::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_mul_runtime!(TensorWithRuntime<L, M, RT>, TensorWithRuntime<R, M, RT>);
impl_mul_runtime!(&'l TensorWithRuntime<L, M, RT>, TensorWithRuntime<R, M, RT>,'l);
impl_mul_runtime!(&'l mut TensorWithRuntime<L, M, RT>, TensorWithRuntime<R, M, RT>,'l);
impl_mul_runtime!(TensorWithRuntime<L, M, RT>, &'r TensorWithRuntime<R, M, RT>,'r);
impl_mul_runtime!(&'l TensorWithRuntime<L, M, RT>, &'r TensorWithRuntime<R, M, RT>,'l,'r);
impl_mul_runtime!(&'l mut TensorWithRuntime<L, M, RT>, &'r TensorWithRuntime<R, M, RT>,'l,'r);
impl_mul_runtime!(TensorWithRuntime<L, M, RT>, &'r mut TensorWithRuntime<R, M, RT>,'r);
impl_mul_runtime!(&'l TensorWithRuntime<L, M, RT>, &'r mut TensorWithRuntime<R, M, RT>,'l,'r);
impl_mul_runtime!(&'l mut TensorWithRuntime<L, M, RT>, &'r mut TensorWithRuntime<R, M, RT>,'l,'r);
