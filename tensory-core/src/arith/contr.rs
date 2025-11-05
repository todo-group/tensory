use core::ops::Mul;

use crate::{
    bound_tensor::{BoundTensor, RuntimeError},
    mapper::{AxisMapper, ConnectAxisOrigin, ConnectMapper},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of contraction operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor must have the proper "axis structure" inherited from the input tensors describe with `axis_origin`.
pub unsafe trait MulCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs contraction operation on the tensors `lhs` and `rhs` with the given axis pairs.
    ///
    /// # Safety
    ///
    /// the user MUST ensure that `axis_origin` has the same numbers of axes same as the input tensors.
    unsafe fn mul_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err>;
}

/// Safe version of `MulCtxImpl`.
///
/// The blanket implementation checks input and panic if the condition is not satisfied.
pub trait MulCtx<Lhs: TensorRepr, Rhs: TensorRepr>: MulCtxImpl<Lhs, Rhs> {
    /// Safe version of `mul_unchecked`.
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
/// Intermediate task struct for contraction operation.
pub struct TensorMul<L: TensorRepr, R: TensorRepr, M: AxisMapper> {
    lhs: L,
    rhs: R,
    res_mapper: M,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TensorMul<L, R, M> {
    /// Construct a `TensorMul` by provided closure.
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
    /// Try to construct a `TensorMul` by provided closure.
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
}

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper, C: MulCtxImpl<L, R>> TensorTask<C>
    for TensorMul<L, R, M>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        //println!("lhs: {:?}, rhs: {:?}", lhs_legs, rhs_legs);
        //println!("idx_pairs: {:?}", idx_pairs);
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                ctx.mul_unchecked(self.lhs, self.rhs, self.axis_origin)?,
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

/// Runtime trait for contraction operation.
pub trait MulRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
    /// The context type.
    type Ctx: MulCtxImpl<Lhs, Rhs>;
    /// Returns the context.
    fn mul_ctx(&self) -> Self::Ctx;
}

// // 9 combinations of Lhs/Rhs being owned/view/view_mut
use crate::bound_tensor::{Runtime, ToBoundTensor};

macro_rules! impl_mul_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>, RT:Runtime> Mul<$r> for $l
        where
            $l: ToBoundTensor<Mapper = M, Runtime = RT>,
            $r: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: MulRuntime<<$l as ToBoundTensor>::Repr, <$r as ToBoundTensor>::Repr>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as MulRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as MulCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    <M as ConnectMapper<2>>::Err,
                    <<RT as MulRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as MulCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Err,
                >,
            >;
            fn mul(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeError::Runtime);
                }
                let res = (lhs * rhs)
                    .map_err(RuntimeError::Axis)?
                    .with(lhs_rt.mul_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_mul_runtime!(BoundTensor<L, M, RT>, BoundTensor<R, M, RT>);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_mul_runtime!(BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'r);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_mul_runtime!(BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'r);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
