use core::ops::Add;

use crate::{
    bound_tensor::{BoundTensor, RuntimeError},
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask},
};

/// Raw context of addition operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the proper "axis structure" inherited from the input tensors described with `axis_mapping`.
pub unsafe trait AddCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
    ///
    /// # Safety
    ///
    /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
    unsafe fn add_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
/// Safe version if `AddCtxImpl`.
///
/// The blanket implementation checks input and panic if the condition is not satisfied.
pub trait AddCtx<Lhs: TensorRepr, Rhs: TensorRepr>: AddCtxImpl<Lhs, Rhs> {
    /// Safe version of `add_unchecked`.
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
        let n = axis_mapping.naxes();
        if n_l != n || n_r != n {
            panic!("axis_mapping must match the number of axes with lhs and rhs");
        }
        unsafe { self.add_unchecked(lhs, rhs, axis_mapping) }
    }
}

/// Intermediate task struct for addition operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TensorAdd<L: TensorRepr, R: TensorRepr, M: AxisMapper> {
    lhs: L,
    rhs: R,
    res_mapper: M,
    axis_mapping: OverlayAxisMapping<2>,
}
impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TensorAdd<L, R, M> {
    /// Construct a `TensorAdd` by provided closure.
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
    /// Try to construct a `TensorAdd` by provided closure.
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

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper, C: AddCtxImpl<L, R>> TensorTask<C>
    for TensorAdd<L, R, M>
{
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Result<Tensor<C::Res, M>, C::Err> {
        Ok(unsafe {
            Tensor::from_raw_unchecked(
                ctx.add_unchecked(self.lhs, self.rhs, self.axis_mapping)?,
                self.res_mapper,
            )
        })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

use crate::tensor::ToTensor;

macro_rules! impl_add {
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

impl_add!(Tensor<L, M>, Tensor<R, M>);
impl_add!(&'l Tensor<L, M>, Tensor<R, M>,'l);
impl_add!(&'l mut Tensor<L, M>, Tensor<R, M>,'l);
impl_add!(Tensor<L, M>, &'r Tensor<R, M>,'r);
impl_add!(&'l Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_add!(&'l mut Tensor<L, M>, &'r Tensor<R, M>,'l,'r);
impl_add!(Tensor<L, M>, &'r mut Tensor<R, M>,'r);
impl_add!(&'l Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);
impl_add!(&'l mut Tensor<L, M>, &'r mut Tensor<R, M>,'l,'r);

/// Runtime trait for addition operation.
pub trait AddRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
    /// The context type.
    type Ctx: AddCtxImpl<Lhs, Rhs>;
    /// Returns the context.
    fn add_ctx(&self) -> Self::Ctx;
}

// // 9 combinations of Lhs/Rhs being owned/view/view_mut
use crate::bound_tensor::{Runtime, ToBoundTensor};

macro_rules! impl_add_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT:Runtime> Add<$r> for $l
        where
            $l: ToBoundTensor<Mapper = M, Runtime = RT>,
            $r: ToBoundTensor<Mapper = M, Runtime = RT>,
            RT: AddRuntime<<$l as ToBoundTensor>::Repr, <$r as ToBoundTensor>::Repr>,
        {
            type Output = Result<
                BoundTensor<
                    <<RT as AddRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as AddCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Res,
                    M,
                    RT,
                >,
                RuntimeError<
                    <M as OverlayMapper<2>>::Err,
                    <<RT as AddRuntime<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Ctx as AddCtxImpl<
                        <$l as ToBoundTensor>::Repr,
                        <$r as ToBoundTensor>::Repr,
                    >>::Err,
                >,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeError::Runtime);
                }
                let res = (lhs + rhs)
                    .map_err(RuntimeError::Axis)?
                    .with(lhs_rt.add_ctx())
                    .map_err(RuntimeError::Ctx)?;
                Ok(BoundTensor::from_raw(res, lhs_rt))
            }
        }
    };
}

impl_add_runtime!(BoundTensor<L, M, RT>, BoundTensor<R, M, RT>);
impl_add_runtime!(&'l BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_add_runtime!(&'l mut BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
impl_add_runtime!(BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'r);
impl_add_runtime!(&'l BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_add_runtime!(&'l mut BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
impl_add_runtime!(BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'r);
impl_add_runtime!(&'l BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
impl_add_runtime!(&'l mut BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
