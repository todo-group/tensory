use core::ops::Mul;

use crate::{
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::{AxisMapper, ConnectAxisOrigin, ConnectMapper},
    port::PortError,
    repr::{TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

// /// Raw context of contraction operation.
// ///
// /// # Safety
// ///
// /// The implementor MUST ensure that the result tensor must have the proper "axis structure" inherited from the input tensors describe with `axis_origin`.
// pub unsafe trait MulCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
//     /// The type of the result tensor representation.
//     type Res: TensorRepr;
//     /// The type of the error returned by the context. (considered as internal error)
//     type Err;

//     /// Performs contraction operation on the tensors `lhs` and `rhs` with the given axis pairs.
//     ///
//     /// # Safety
//     ///
//     /// the user MUST ensure that `axis_origin` has the same numbers of axes same as the input tensors.
//     unsafe fn mul_unchecked(
//         self,
//         lhs: Lhs,
//         rhs: Rhs,
//         axis_origin: ConnectAxisOrigin<2>,
//     ) -> Result<Self::Res, Self::Err>;
// }

// /// Safe version of `MulCtxImpl`.
// ///
// /// The blanket implementation checks input and panic if the condition is not satisfied.
// pub trait MulCtx<Lhs: TensorRepr, Rhs: TensorRepr>: MulCtxImpl<Lhs, Rhs> {
//     /// Safe version of `mul_unchecked`.
//     fn mul(
//         self,
//         lhs: Lhs,
//         rhs: Rhs,
//         axis_origin: ConnectAxisOrigin<2>,
//     ) -> Result<Self::Res, Self::Err>;
// }
// impl<C: MulCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> MulCtx<Lhs, Rhs> for C {
//     fn mul(
//         self,
//         lhs: Lhs,
//         rhs: Rhs,
//         axis_origin: ConnectAxisOrigin<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         if axis_origin.in_lens() != [lhs.naxes(), rhs.naxes()] {
//             panic!("axis_origin must match the number of axes with lhs and rhs");
//         }

//         unsafe { self.mul_unchecked(lhs, rhs, axis_origin) }
//     }
// }

/// Intermediate task struct for contraction operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MulRepr<L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorRepr, R: TensorRepr> MulRepr<L, R> {
    /// Construct a `MulRepr` by provided closure.
    pub unsafe fn from_raw_unchecked(lhs: L, rhs: R, axis_origin: ConnectAxisOrigin<2>) -> Self {
        Self {
            lhs,
            rhs,
            axis_origin,
        }
    }
    pub fn from_raw(
        self,
        lhs: L,
        rhs: R,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self, PortError> {
        let n_l = lhs.naxes();
        let n_r = rhs.naxes();
        let n = axis_origin.len();
        if n_l != n || n_r != n {
            Err(PortError)
        } else {
            Ok(Self {
                lhs,
                rhs,
                axis_origin,
            })
        }
    }

    pub fn into_raw(self) -> (L, R, ConnectAxisOrigin<2>) {
        (self.lhs, self.rhs, self.axis_origin)
    }
}

unsafe impl<L: TensorRepr, R: TensorRepr> TensorTupleRepr<1> for MulRepr<L, R> {
    fn naxeses(&self) -> [usize; 1] {
        [self.axis_origin.len()]
    }
}

impl<L: TensorRepr, R: TensorRepr> IsTask for MulRepr<L, R> {}

impl<L: TensorRepr, M: AxisMapper> Tensor<L, M> {
    // /// Construct a `TensorAdd` by provided closure.
    pub unsafe fn mul_by_manager_unchecked<
        R: TensorRepr,
        C: ContainerMapImpl<(M, ConnectAxisOrigin<2>), Tensor<MulRepr<L, R>, M>>,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, ConnectAxisOrigin<2>)>>::Container,
    ) -> <C as ContainerImpl<Tensor<MulRepr<L, R>, M>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();

        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_origin)| unsafe {
                Tensor::from_raw_unchecked(
                    MulRepr {
                        lhs,
                        rhs,
                        axis_origin,
                    },
                    [res_mapper],
                )
            },
        )
    }
    // /// Try to construct a `TensorAdd` by provided closure.
    pub fn mul_by_manager_checked<
        R: TensorRepr,
        C: ContainerMapImpl<
                (M, ConnectAxisOrigin<2>),
                Result<Tensor<MulRepr<L, R>, M>, (MulRepr<L, R>, M)>,
            >,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, ConnectAxisOrigin<2>)>>::Container,
    ) -> <C as ContainerImpl<Result<Tensor<MulRepr<L, R>, M>, (MulRepr<L, R>, M)>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();

        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_origin)| {
                Tensor::from_raw(
                    MulRepr {
                        lhs,
                        rhs,
                        axis_origin,
                    },
                    [res_mapper],
                )
                .map_err(|(r, [m])| (r, m))
            },
        )
    }
}

// pub unsafe trait TensorMulContext<Mk, L: TensorRepr, R: TensorRepr, O>:
//     Context<Mk, MulRepr<L, R>, O>
// {
//     // type Ctx: Context<Mk, AddRepr<L, R>, O>;
//     // fn add_ctx(&self) -> Self::Ctx;
// }

// impl<L: TensorRepr, R: TensorRepr, M: AxisMapper> TaskHolder<MulRepr<L, R>>
//     for Tensor<MulRepr<L, R>, M>
// {
// }

// impl<
//     L: TensorRepr,
//     R: TensorRepr,
//     M: AxisMapper,
//     Mk,
//     Ctx: TensorMulContext<Mk, L, R, O>,
//     O: TensorRepr,
// > TaskDelegate<MulRepr<L, R>, O, Mk, Ctx> for Tensor<MulRepr<L, R>, M>
// {
//     type Output = Tensor<O, M>;

//     fn with(self, ctx: Ctx) -> Self::Output {
//         let (repr, mapper) = self.into_raw();
//         let output = ctx.execute(repr);

//         unsafe { Tensor::from_raw_unchecked(output, mapper) }
//     }
// }
// impl<
//     L: TensorRepr,
//     R: TensorRepr,
//     M: AxisMapper,
//     Mk,
//     Ctx: TensorMulContext<Mk, L, R, Result<Ores, Oerr>>,
//     Ores: TensorRepr,
//     Oerr,
// > TaskDelegate<MulRepr<L, R>, Result<Ores, Oerr>, Mk, Ctx> for Tensor<MulRepr<L, R>, M>
// {
//     type Output = Result<Tensor<Ores, M>, Oerr>;

//     fn with(self, ctx: Ctx) -> Self::Output {
//         let (repr, mapper) = self.into_raw();
//         let output = ctx.execute(repr)?;

//         Ok(unsafe { Tensor::from_raw_unchecked(output, mapper) })
//     }
// }

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_mul {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>> Mul<$r> for $l
        where
            $l: ToTensorTuple<1,Mapper = M>,
            $r: ToTensorTuple<1,Mapper = M>,
            M::CType: ContainerMapImpl<(M, ConnectAxisOrigin<2>), Tensor<MulRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>,
        {
            type Output = <M::CType as ContainerImpl<Tensor<MulRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>>::Container;

            fn mul(self, rhs: $r) -> Self::Output {
                let lhs = ToTensorTuple::<1>::to_tensor_tuple(self);
                let rhs = ToTensorTuple::<1>::to_tensor_tuple(rhs);
                unsafe { lhs.mul_by_manager_unchecked::<_, M::CType>(rhs, |l, r| ConnectMapper::<2>::connect([l, r])) }
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

// /// Runtime trait for contraction operation.
// pub trait MulRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
//     /// The context type.
//     type Ctx: MulCtxImpl<Lhs, Rhs>;
//     /// Returns the context.
//     fn mul_ctx(&self) -> Self::Ctx;
// }

// // // 9 combinations of Lhs/Rhs being owned/view/view_mut
// use crate::bound_tensor::{Runtime, ToBoundTensor};

macro_rules! impl_mul_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>, RT:Runtime> Mul<$r> for $l
        where
            $l: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            $r: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M>,
                BoundTensorTuple<1, <RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
            >,
            M::CType: ContainerMapImpl<(M, ConnectAxisOrigin<2>), Tensor<MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>
            + ContainerMapImpl<
                Tensor<MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                <<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                    BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                >>::Container
            >,
        {
            type Output = Result<
                <M::CType as ContainerImpl<
                    <<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::Container,
            PortError>;

            fn mul(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();

                let res =<M::CType as ContainerMapImpl<
                    Tensor<MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                    <<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::map(lhs * rhs,|mul_t|{
                    let res=lhs_rt.ctx().execute(mul_t);
                    <RT::Ctx as TensorContext<RT::Mk, 1, MulRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType::map(res, |res| {
                        BoundTensorTuple::from_raw(res, lhs_rt)
                    })
                });
                Ok(res)

                // if lhs_rt != rhs_rt {
                //     return Err(PortError);
                // }
                // let res = (lhs * rhs)
                //     .map_err(RuntimeErr::Axis)?
                //     .with(lhs_rt.mul_ctx())
                //     .map_err(RuntimeErr::Ctx)?;
                // Ok(BoundTensor::from_raw(res, lhs_rt))
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
