use core::ops::Mul;

use crate::{
    mapper::{AxisMapper, ConnectAxisOrigin, ConnectMapper},
    repr::TensorRepr,
    tensor::{Tensor, ToTensor},
    tensor_with_runtime::{RuntimeError, TensorWithRuntime},
};

/// Raw context of contraction operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor must have the proper axis structure.
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

/// Safe version of ContractionContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait MulCtx<Lhs: TensorRepr, Rhs: TensorRepr>: MulCtxImpl<Lhs, Rhs> {
    /// Safe version of mul_unchecked.
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
        if axis_origin.in_lens() != [lhs.naxes(), rhs.naxes()] {
            panic!("axis_origin must match the number of axes with lhs and rhs");
        }

        unsafe { self.mul_unchecked(lhs, rhs, axis_origin) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMul<L: TensorRepr, R: TensorRepr, M: AxisMapper> {
    lhs: L,
    rhs: R,
    res_mapper: M,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TensorMul<L, R, M> {
    pub fn by_manager(
        lhs: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> (M, ConnectAxisOrigin<2>),
    ) -> Self {
        let (lhs, lhs_mapper) = lhs.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_origin) = manager(lhs_mapper, rhs_mapper);

        Self {
            lhs,
            rhs,
            res_mapper,
            axis_origin,
        }
    }
    pub fn try_by_manager<E>(
        lhs: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> Result<(M, ConnectAxisOrigin<2>), E>,
    ) -> Result<Self, E> {
        let (lhs, lhs_mapper) = lhs.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_origin) = manager(lhs_mapper, rhs_mapper)?;

        Ok(Self {
            lhs,
            rhs,
            res_mapper,
            axis_origin,
        })
    }

    pub fn with<C: MulCtx<L, R>>(self, context: C) -> Result<Tensor<C::Res, M>, C::Err> {
        //println!("lhs: {:?}, rhs: {:?}", lhs_legs, rhs_legs);
        //println!("idx_pairs: {:?}", idx_pairs);

        Ok(unsafe {
            Tensor::from_raw_unchecked(
                context.mul_unchecked(self.lhs, self.rhs, self.axis_origin)?,
                self.res_mapper,
            )
        })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_mul {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>> Mul<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                TensorMul<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr, M>,
                <M as ConnectMapper<2>>::Err,
            >;
            fn mul(self, rhs: $r) -> Self::Output {
                let lhs = ToTensor::to_tensor(self);
                let rhs = ToTensor::to_tensor(rhs);
                TensorMul::try_by_manager(lhs, rhs, |l, r| ConnectMapper::<2>::connect([l, r]))
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

pub trait MulRuntime<Lhs: TensorRepr, Rhs: TensorRepr> {
    type Ctx: MulCtxImpl<Lhs, Rhs>;
    fn mul_ctx(self) -> Self::Ctx;
}

// // 9 combinations of Lhs/Rhs being owned/view/view_mut
use crate::tensor_with_runtime::ToTensorWithRuntime;

macro_rules! impl_mul_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>, RT> Mul<$r> for $l
        where
            $l: ToTensorWithRuntime<Mapper = M, Runtime = RT>,
            $r: ToTensorWithRuntime<Mapper = M, Runtime = RT>,
            RT: Copy + Eq + MulRuntime<<$l as ToTensorWithRuntime>::Repr, <$r as ToTensorWithRuntime>::Repr>,
        {
            type Output = Result<
                TensorWithRuntime<
                    <<RT as MulRuntime<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Ctx as MulCtxImpl<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    <M as ConnectMapper<2>>::Err,
                    <<RT as MulRuntime<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Ctx as MulCtxImpl<
                        <$l as ToTensorWithRuntime>::Repr,
                        <$r as ToTensorWithRuntime>::Repr,
                    >>::Err,
                >,
            >;
            fn mul(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_tensor_with_runtime().into_raw();
                let (rhs, rhs_rt) = rhs.to_tensor_with_runtime().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeError::Runtime);
                }
                let res = (lhs * rhs)
                    .map_err(RuntimeError::Axis)?
                    .with(lhs_rt.mul_ctx())
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
