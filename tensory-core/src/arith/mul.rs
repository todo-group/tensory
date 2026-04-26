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

/// Intermediate task representation for contraction (multiplication) operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MulRepr<L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorRepr, R: TensorRepr> MulRepr<L, R> {
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
        if axis_origin.in_lens() != [lhs.naxes(), rhs.naxes()] {
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
    /// Construct a contraction task tensor by provided closure with unchecked manager construction.
    /// # Safety
    /// The caller MUST ensure that the manager returns a consistent axis mapper and mapping.
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
    /// Construct a contraction task tensor by provided closure with checked manager construction.
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

// 9 combinations of Lhs/Rhs being owned/view/view_mut
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
                if lhs_rt != rhs_rt {
                    return Err(PortError);
                }
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
