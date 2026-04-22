use core::ops::Sub;

use crate::{
    bound_tensor::{BoundTensorTuple, RuntimeErr},
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    port::PortError,
    repr::TensorRepr,
    task::{Context, TaskDelegate, TaskHolder},
    tensor::{Tensor, ToTensorTuple},
};

/// Intermediate task struct for subtraction operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SubRepr<L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    axis_mapping: OverlayAxisMapping<2>,
}

impl<L: TensorRepr, R: TensorRepr> SubRepr<L, R> {
    pub unsafe fn from_raw_unchecked(lhs: L, rhs: R, axis_mapping: OverlayAxisMapping<2>) -> Self {
        Self {
            lhs,
            rhs,
            axis_mapping,
        }
    }
    pub fn from_raw(
        self,
        lhs: L,
        rhs: R,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self, PortError> {
        let n_l = lhs.naxes();
        let n_r = rhs.naxes();
        let n = axis_mapping.naxes();
        if n_l != n || n_r != n {
            Err(PortError)
        } else {
            Ok(Self {
                lhs,
                rhs,
                axis_mapping,
            })
        }
    }

    pub fn into_raw(self) -> (L, R, OverlayAxisMapping<2>) {
        (self.lhs, self.rhs, self.axis_mapping)
    }
}

unsafe impl<L: TensorRepr, R: TensorRepr> TensorRepr for SubRepr<L, R> {
    fn naxes(&self) -> usize {
        self.axis_mapping.naxes()
    }
}

impl<L: TensorRepr, M: AxisMapper> Tensor<L, M> {
    // /// Construct a `TensorAdd` by provided closure.
    pub fn sub_by_manager<R: TensorRepr>(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> (M, OverlayAxisMapping<2>),
    ) -> Tensor<SubRepr<L, R>, M> {
        let (lhs, lhs_mapper) = self.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_mapping) = manager(lhs_mapper, rhs_mapper);

        unsafe {
            Tensor::from_raw_unchecked(
                SubRepr {
                    lhs,
                    rhs,
                    axis_mapping,
                },
                res_mapper,
            )
        }
    }
    // /// Try to construct a `TensorAdd` by provided closure.
    pub fn try_sub_by_manager<R: TensorRepr, E>(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> Result<(M, OverlayAxisMapping<2>), E>,
    ) -> Result<Tensor<SubRepr<L, R>, M>, E> {
        let (lhs, lhs_mapper) = self.into_raw();
        let (rhs, rhs_mapper) = rhs.into_raw();

        let (res_mapper, axis_origin) = manager(lhs_mapper, rhs_mapper)?;

        Ok(unsafe {
            Tensor::from_raw_unchecked(
                SubRepr {
                    lhs,
                    rhs,
                    axis_mapping: axis_origin,
                },
                res_mapper,
            )
        })
    }
}

pub unsafe trait TensorSubContext<Mk, L: TensorRepr, R: TensorRepr, O>:
    Context<Mk, SubRepr<L, R>, O>
{
    // type Ctx: Context<Mk, AddRepr<L, R>, O>;
    // fn add_ctx(&self) -> Self::Ctx;
}

impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TaskHolder<SubRepr<L, R>>
    for Tensor<SubRepr<L, R>, M>
{
}

impl<
    L: TensorRepr,
    R: TensorRepr,
    M: AxisMapper,
    Mk,
    Ctx: TensorSubContext<Mk, L, R, O>,
    O: TensorRepr,
> TaskDelegate<SubRepr<L, R>, O, Mk, Ctx> for Tensor<SubRepr<L, R>, M>
{
    type Output = Tensor<O, M>;

    fn with(self, ctx: Ctx) -> Self::Output {
        let (repr, mapper) = self.into_raw();
        let output = ctx.execute(repr);

        unsafe { Tensor::from_raw_unchecked(output, mapper) }
    }
}
impl<
    L: TensorRepr,
    R: TensorRepr,
    M: AxisMapper,
    Mk,
    Ctx: TensorSubContext<Mk, L, R, Result<Ores, Oerr>>,
    Ores: TensorRepr,
    Oerr,
> TaskDelegate<SubRepr<L, R>, Result<Ores, Oerr>, Mk, Ctx> for Tensor<SubRepr<L, R>, M>
{
    type Output = Result<Tensor<Ores, M>, Oerr>;

    fn with(self, ctx: Ctx) -> Self::Output {
        let (repr, mapper) = self.into_raw();
        let output = ctx.execute(repr)?;

        Ok(unsafe { Tensor::from_raw_unchecked(output, mapper) })
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_sub {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>> Sub<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                Tensor<SubRepr<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr>, M>,
                <M as OverlayMapper<2>>::Err,
            >;
            fn sub(self, rhs: $r) -> Self::Output {
                let lhs = ToTensor::to_tensor(self);
                let rhs = ToTensor::to_tensor(rhs);
                lhs.try_sub_by_manager(rhs, |l, r| OverlayMapper::<2>::overlay([l, r]))
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

// /// Runtime trait for subtraction operation.
// pub trait SubRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
//     /// The context type.
//     type Ctx: SubCtxImpl<Lhs, Rhs>;
//     /// Returns the context.
//     fn sub_ctx(&self) -> Self::Ctx;
// }

// // // 9 combinations of Lhs/Rhs being owned/view/view_mut
// use crate::bound_tensor::{Runtime, ToBoundTensor};

// macro_rules! impl_sub_runtime {
//     ($l:ty,$r:ty $(,$life:lifetime)*) => {
//         impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT:Runtime> Sub<$r> for $l
//         where
//             $l: ToBoundTensor<Mapper = M, Runtime = RT>,
//             $r: ToBoundTensor<Mapper = M, Runtime = RT>,
//             RT: SubRuntime<<$l as ToBoundTensor>::Repr, <$r as ToBoundTensor>::Repr>,
//         {
//             type Output = Result<
//                 BoundTensor<
//                     <<RT as SubRuntime<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Ctx as SubCtxImpl<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Res,
//                     M,
//                     RT,
//                 >,
//                 RuntimeErr<
//                     <M as OverlayMapper<2>>::Err,
//                     <<RT as SubRuntime<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Ctx as SubCtxImpl<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Err,
//                 >,
//             >;
//             fn sub(self, rhs: $r) -> Self::Output {
//                 let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();
//                 let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

//                 if lhs_rt != rhs_rt {
//                     return Err(RuntimeErr::Runtime);
//                 }
//                 let res = (lhs - rhs)
//                     .map_err(RuntimeErr::Axis)?
//                     .with(lhs_rt.sub_ctx())
//                     .map_err(RuntimeErr::Ctx)?;
//                 Ok(BoundTensor::from_raw(res, lhs_rt))
//             }
//         }
//     };
// }

// impl_sub_runtime!(BoundTensor<L, M, RT>, BoundTensor<R, M, RT>);
// impl_sub_runtime!(&'l BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
// impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, BoundTensor<R, M, RT>,'l);
// impl_sub_runtime!(BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'r);
// impl_sub_runtime!(&'l BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
// impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>,'l,'r);
// impl_sub_runtime!(BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'r);
// impl_sub_runtime!(&'l BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
// impl_sub_runtime!(&'l mut BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>,'l,'r);
