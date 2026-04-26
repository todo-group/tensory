use core::ops::Sub;

use crate::{
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    port::PortError,
    repr::{TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

/// Intermediate task representation for subtraction operation.
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

unsafe impl<L: TensorRepr, R: TensorRepr> TensorTupleRepr<1> for SubRepr<L, R> {
    fn naxeses(&self) -> [usize; 1] {
        [self.axis_mapping.naxes()]
    }
}
impl<L: TensorRepr, R: TensorRepr> IsTask for SubRepr<L, R> {}

impl<L: TensorRepr, M: AxisMapper> Tensor<L, M> {
    /// Construct a subtraction task tensor by provided closure with unchecked manager construction.
    /// # Safety
    /// The caller MUST ensure that the manager returns a consistent axis mapper and mapping.
    pub unsafe fn sub_by_manager_unchecked<
        R: TensorRepr,
        C: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<SubRepr<L, R>, M>>,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, OverlayAxisMapping<2>)>>::Container,
    ) -> <C as ContainerImpl<Tensor<SubRepr<L, R>, M>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();
        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_mapping)| unsafe {
                Tensor::from_raw_unchecked(
                    SubRepr {
                        lhs,
                        rhs,
                        axis_mapping,
                    },
                    [res_mapper],
                )
            },
        )
    }
    /// Construct a subtraction task tensor by provided closure with checked manager construction.
    pub fn sub_by_manager_checked<
        R: TensorRepr,
        C: ContainerMapImpl<
                (M, OverlayAxisMapping<2>),
                Result<Tensor<SubRepr<L, R>, M>, (SubRepr<L, R>, M)>,
            >,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, OverlayAxisMapping<2>)>>::Container,
    ) -> <C as ContainerImpl<Result<Tensor<SubRepr<L, R>, M>, (SubRepr<L, R>, M)>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();
        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_mapping)| {
                Tensor::from_raw(
                    SubRepr {
                        lhs,
                        rhs,
                        axis_mapping,
                    },
                    [res_mapper],
                )
                .map_err(|(r, [m])| (r, m))
            },
        )
    }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut
macro_rules! impl_sub {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>> Sub<$r> for $l
        where
            $l: ToTensorTuple<1,Mapper = M>,
            $r: ToTensorTuple<1,Mapper = M>,
            M::CType: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<SubRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>,
        {
            type Output = <M::CType as ContainerImpl<Tensor<SubRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>>::Container;
            fn sub(self, rhs: $r) -> Self::Output {
                let lhs = ToTensorTuple::<1>::to_tensor_tuple(self);
                let rhs = ToTensorTuple::<1>::to_tensor_tuple(rhs);
                unsafe { lhs.sub_by_manager_unchecked::<_, M::CType>(rhs, |l, r| OverlayMapper::<2>::overlay([l, r])) }
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

// 9 combinations of Lhs/Rhs being owned/view/view_mut
macro_rules! impl_sub_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT:Runtime> Sub<$r> for $l
        where
            $l: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            $r: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
            >,
            M::CType: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>
            + ContainerMapImpl<
                Tensor<SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                <<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                    BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                >>::Container
            >,
        {
            type Output = Result<
                <M::CType as ContainerImpl<
                    <<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::Container,
            PortError>;

            fn sub(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(PortError);
                }
                let res =<M::CType as ContainerMapImpl<
                    Tensor<SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                    <<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::map(lhs - rhs,|sub_t|{
                    let res=lhs_rt.ctx().execute(sub_t);
                    <RT::Ctx as TensorContext<RT::Mk, 1, SubRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType::map(res, |res| {
                        BoundTensorTuple::from_raw(res, lhs_rt)
                    })
                });
                Ok(res)
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
