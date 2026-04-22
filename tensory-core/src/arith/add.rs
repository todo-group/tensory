use core::ops::Add;

use crate::{
    bound_tensor::{BoundTensor, BoundTensorTuple, Runtime, RuntimeImpl, ToBoundTensorTuple},
    container::{ContainerImpl, ContainerMapImpl},
    mapper::{AxisMapper, OverlayAxisMapping, OverlayMapper},
    port::PortError,
    repr::{TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
    tensor::{Tensor, TensorContext, ToTensorTuple},
};

/// Intermediate task struct for addition operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AddRepr<L: TensorRepr, R: TensorRepr> {
    lhs: L,
    rhs: R,
    axis_mapping: OverlayAxisMapping<2>,
}
impl<L: TensorRepr, R: TensorRepr> AddRepr<L, R> {
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

unsafe impl<L: TensorRepr, R: TensorRepr> TensorTupleRepr<1> for AddRepr<L, R> {
    fn naxeses(&self) -> [usize; 1] {
        [self.axis_mapping.naxes()]
    }
}
impl<L: TensorRepr, R: TensorRepr> IsTask for AddRepr<L, R> {}

impl<L: TensorRepr, M: AxisMapper> Tensor<L, M> {
    // /// Construct a `TensorAdd` by provided closure.
    pub unsafe fn add_by_manager_unchecked<
        R: TensorRepr,
        C: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<AddRepr<L, R>, M>>,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, OverlayAxisMapping<2>)>>::Container,
    ) -> <C as ContainerImpl<Tensor<AddRepr<L, R>, M>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();
        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_mapping)| unsafe {
                Tensor::from_raw_unchecked(
                    AddRepr {
                        lhs,
                        rhs,
                        axis_mapping,
                    },
                    [res_mapper],
                )
            },
        )
    }

    pub fn add_by_manager_checked<
        R: TensorRepr,
        C: ContainerMapImpl<
                (M, OverlayAxisMapping<2>),
                Result<Tensor<AddRepr<L, R>, M>, (AddRepr<L, R>, M)>,
            >,
    >(
        self: Tensor<L, M>,
        rhs: Tensor<R, M>,
        manager: impl FnOnce(M, M) -> <C as ContainerImpl<(M, OverlayAxisMapping<2>)>>::Container,
    ) -> <C as ContainerImpl<Result<Tensor<AddRepr<L, R>, M>, (AddRepr<L, R>, M)>>>::Container {
        let (lhs, [lhs_mapper]) = self.into_raw();
        let (rhs, [rhs_mapper]) = rhs.into_raw();
        C::map(
            manager(lhs_mapper, rhs_mapper),
            |(res_mapper, axis_mapping)| {
                Tensor::from_raw(
                    AddRepr {
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
    // // /// Try to construct a `TensorAdd` by provided closure.
    // pub fn try_add_by_manager<R: TensorRepr, E>(
    //     self: Tensor<L, M>,
    //     rhs: Tensor<R, M>,
    //     manager: impl FnOnce(M, M) -> Result<(M, OverlayAxisMapping<2>), E>,
    // ) -> Result<Tensor<AddRepr<L, R>, M>, E> {
    //     let (lhs, [lhs_mapper]) = self.into_raw();
    //     let (rhs, [rhs_mapper]) = rhs.into_raw();

    //     let (res_mapper, axis_origin) = manager(lhs_mapper, rhs_mapper)?;

    //     Ok(unsafe {
    //         Tensor::from_raw_unchecked(
    //             AddRepr {
    //                 lhs,
    //                 rhs,
    //                 axis_mapping: axis_origin,
    //             },
    //             [res_mapper],
    //         )
    //     })
    // }
}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_add {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>> Add<$r> for $l
        where
            $l: ToTensorTuple<1,Mapper = M>,
            $r: ToTensorTuple<1,Mapper = M>,
            M::CType: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<AddRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>,
        {
            type Output = <M::CType as ContainerImpl<Tensor<AddRepr<<$l as ToTensorTuple<1>>::Repr, <$r as ToTensorTuple<1>>::Repr>, M>>>::Container;
            fn add(self, rhs: $r) -> Self::Output {
                let lhs = ToTensorTuple::<1>::to_tensor_tuple(self);
                let rhs = ToTensorTuple::<1>::to_tensor_tuple(rhs);
                unsafe { lhs.add_by_manager_unchecked::<_, M::CType>(rhs, |l, r| OverlayMapper::<2>::overlay([l, r])) }
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

// /// Runtime trait for addition operation.
// pub trait AddRuntime<Lhs: TensorRepr, Rhs: TensorRepr>: Runtime {
//     /// The context type.
//     type Mk;
//     type Ctx: TensorContext<Self::Mk, 1, AddRepr<Lhs, Rhs>>;
//     /// Returns the context.
//     fn add_ctx(&self) -> Self::Ctx;
// }

// // 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_add_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorRepr, R: TensorRepr, M: OverlayMapper<2>, RT:Runtime> Add<$r> for $l
        where
            $l: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            $r: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeImpl<Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>,
            RT::Ctx: TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
            <RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType: ContainerMapImpl<
                Tensor<<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M>,
                BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
            >,
            M::CType: ContainerMapImpl<(M, OverlayAxisMapping<2>), Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>
            + ContainerMapImpl<
                Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                <<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                    BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                >>::Container
            >,
        {
            type Output = Result<
                <M::CType as ContainerImpl<
                    <<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::Container,
            PortError>;

            fn add(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(PortError);
                }
                let res =<M::CType as ContainerMapImpl<
                    Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>,
                    <<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType as ContainerImpl<
                        BoundTensor<<RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,M,RT>,
                    >>::Container
                >>::map(lhs + rhs,|add_t|{
                    let res=lhs_rt.ctx().execute(add_t);
                    <RT::Ctx as TensorContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::CType::map(res, |res| {
                        BoundTensorTuple::from_raw(res, lhs_rt)
                    })
                });
                Ok(res)
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
