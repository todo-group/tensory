use core::ops::Add;

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::{OverlayAxisMapping, OverlayMapper},
    repr::{TensorRepr, TensorTupleRepr},
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple},
};

/// Lazy representation for an addition operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AddRepr<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> {
    lhs: L,
    rhs: R,
    axis_mapping: OverlayAxisMapping<2>,
}

impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> AddRepr<L, R> {
    /// Creates a representation after validating its axis mapping.
    pub fn from_raw(
        lhs: L,
        rhs: R,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self, (L, R, OverlayAxisMapping<2>)> {
        if axis_mapping.naxes() == lhs.naxes() && axis_mapping.naxes() == rhs.naxes() {
            Ok(unsafe { Self::from_raw_unchecked(lhs, rhs, axis_mapping) })
        } else {
            Err((lhs, rhs, axis_mapping))
        }
    }

    /// Creates a representation without checking its axis mapping.
    ///
    /// # Safety
    ///
    /// `axis_mapping` must describe the axes of both inputs.
    pub unsafe fn from_raw_unchecked(lhs: L, rhs: R, axis_mapping: OverlayAxisMapping<2>) -> Self {
        Self {
            lhs,
            rhs,
            axis_mapping,
        }
    }

    /// Decomposes the representation into its inputs and axis mapping.
    pub fn into_raw(self) -> (L, R, OverlayAxisMapping<2>) {
        (self.lhs, self.rhs, self.axis_mapping)
    }
}

unsafe impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> TensorTupleRepr<1> for AddRepr<L, R> {
    fn naxes_array(&self) -> [usize; 1] {
        [self.axis_mapping.naxes()]
    }
}

impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> IsTask for AddRepr<L, R> {}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

use crate::tensor::ToTensor;

macro_rules! impl_add {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorTupleRepr<1>, R: TensorTupleRepr<1>, M: OverlayMapper<2>> Add<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                Tensor<AddRepr<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr>, M>,
                <M as OverlayMapper<2>>::Err,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let (lhs, [lhs_mapper]) = ToTensor::to_tensor(self).into_raw();
                let (rhs, [rhs_mapper]) = ToTensor::to_tensor(rhs).into_raw();
                OverlayMapper::<2>::overlay([lhs_mapper, rhs_mapper]).map(
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

// // 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_add_runtime {
    ($l:ty,$r:ty $(,$life:lifetime)*) => {
        impl<$($life,)* L: TensorTupleRepr<1>, R: TensorTupleRepr<1>, M: OverlayMapper<2> + Clone, RT: IsRuntime, Err
        > Add<$r> for $l
        where
            $l: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            $r: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>,
            <RT as RuntimeFor<Tensor<AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M,
                CType = Resulting<Raw,Err>
            >
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<RT::Mk, 1, AddRepr<<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<
                    <M as OverlayMapper<2>>::Err,
                    Err
                >,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeErr::Runtime);
                }
                let rt= lhs_rt;
                let task = (lhs + rhs).map_err(RuntimeErr::Defer)?;
                let res = rt.ctx().execute(task).map_err(RuntimeErr::Execute)?;

                Ok(BoundTensor::from_raw(res, rt))
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
