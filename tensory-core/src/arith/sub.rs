use core::ops::Sub;

use crate::{
    bound_tensor::{BoundTensor, RuntimeError},
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask},
};

/// Raw context of subtraction operation.
///
/// The implementor MUST ensure that the result tensor must have the proper axis structure specified by `axis_mapping`.
pub unsafe trait SubCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs subtraction operation on the tensors `lhs` and `rhs` with the given axis pairs, and returns the result tensor each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that `axis_origin` is for the number of axes same as the input tensors.
    unsafe fn sub_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
/// Safe version if SubCtxImpl.
///
/// The blanket implementation checks both input and output.
pub trait SubCtx<Lhs: TensorRepr, Rhs: TensorRepr>: SubCtxImpl<Lhs, Rhs> {
    fn sub(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
impl<C: SubCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> SubCtx<Lhs, Rhs> for C {
    fn sub(
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
        unsafe { self.sub_unchecked(lhs, rhs, axis_mapping) }
    }
}

pub struct TensorSub<L: TensorRepr, R: TensorRepr, M: AxisMapper> {
    lhs: L,
    rhs: R,
    res_mapper: M,
    axis_mapping: OverlayAxisMapping<2>,
}
impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TensorSub<L, R, M> {
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
}

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper, C: SubCtxImpl<L, R>> TensorTask<C>
    for TensorSub<L, R, M>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Result<Tensor<C::Res, M>, C::Err> {
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                ctx.sub_unchecked(self.lhs, self.rhs, self.axis_mapping)?,
                self.res_mapper,
            )
        })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

use crate::tensor::ToTensor;

macro_rules! impl_sub {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>> Sub<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                TensorSub<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr, M>,
                <M as OverlayMapper<2>>::Err,
            >;
            fn sub(self, rhs: $r) -> Self::Output {
                let lhs = ToTensor::to_tensor(self);
                let rhs = ToTensor::to_tensor(rhs);
                TensorSub::try_by_manager(lhs, rhs, |l, r| OverlayMapper::<2>::overlay([l, r]))
            }
        }
    };
}

impl_sub!(Tensor<L, M>, Tensor<R, M>);
impl_sub!(&'l Tensor<L, M>, Tensor<R, M>,'l);
impl_sub!(&'l mut Tensor<L, M>, Tensor<R, M>,'l);
impl_sub!(Tensor<L, M>, &'r Tensor<R, M>,'r);
impl_sub!(&'l Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_sub!(&'l mut Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_sub!(Tensor<L, M>, &'r mut Tensor<R, M>,'r);
impl_sub!(&'l Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);
impl_sub!(&'l mut Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);

pub trait SubRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
    type Ctx: SubCtxImpl<Lhs, Rhs>;
    fn sub_ctx(&self) -> Self::Ctx;
}

// // 9 combinations of Lhs/Rhs being owned/view/view_mut
use crate::bound_tensor::{Runtime, ToBoundTensor};

macro_rules! impl_sub_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT:Runtime> Sub<$r> for $l
        where
            $l: ToBoundTensor<Mapper = M, Runtime = RT>,
            $r: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: SubRuntime<<$l as ToBoundTensor>::Repr, <$r as ToBoundTensor>::Repr>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as SubRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as SubCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    <M as OverlayMapper<2>>::Err,
                    <<RT as SubRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as SubCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Err,
                >,
            >;
            fn sub(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeError::Runtime);
                }
                let res = (lhs - rhs)
                    .map_err(RuntimeError::Axis)?
                    .with(lhs_rt.sub_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_sub_runtime!(BoundTensor<L, M, RT>, BoundTensor<R, M, RT>);
impl_sub_runtime!(&'l BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_sub_runtime!(BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'r);
impl_sub_runtime!(&'l BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_sub_runtime!(BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'r);
impl_sub_runtime!(&'l BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
