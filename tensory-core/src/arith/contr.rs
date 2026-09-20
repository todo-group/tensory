use core::ops::Mul;

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::{ConnectAxisOrigin, ConnectMapper},
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Lazy representation for a tensor contraction operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MulRepr<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> {
    lhs: L,
    rhs: R,
    axis_origin: ConnectAxisOrigin<2>,
}

impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> MulRepr<L, R> {
    /// Creates a representation after validating its axis origin.
    pub fn from_raw(
        lhs: L,
        rhs: R,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self, (L, R, ConnectAxisOrigin<2>)> {
        let [lhs_naxes] = lhs.naxes_array();
        let [rhs_naxes] = rhs.naxes_array();
        if axis_origin.in_lens() == [lhs_naxes, rhs_naxes] {
            Ok(unsafe { Self::from_raw_unchecked(lhs, rhs, axis_origin) })
        } else {
            Err((lhs, rhs, axis_origin))
        }
    }

    /// Creates a representation without checking its axis origin.
    ///
    /// # Safety
    ///
    /// `axis_origin` must describe every axis of both inputs.
    pub unsafe fn from_raw_unchecked(lhs: L, rhs: R, axis_origin: ConnectAxisOrigin<2>) -> Self {
        Self {
            lhs,
            rhs,
            axis_origin,
        }
    }

    /// Decomposes the representation into its inputs and axis origin.
    pub fn into_raw(self) -> (L, R, ConnectAxisOrigin<2>) {
        (self.lhs, self.rhs, self.axis_origin)
    }
}

unsafe impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> TensorTupleRepr<1> for MulRepr<L, R> {
    fn naxes_array(&self) -> [usize; 1] {
        [self.axis_origin.len()]
    }
}

impl<L: TensorTupleRepr<1>, R: TensorTupleRepr<1>> IsTask for MulRepr<L, R> {}

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_mul {
    ($l:ty, $r:ty $(, $life:lifetime)*) => {
        impl<$($life,)* L: TensorTupleRepr<1>, R: TensorTupleRepr<1>, M: ConnectMapper<2>>
            Mul<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                Tensor<MulRepr<<$l as ToTensor>::Repr, <$r as ToTensor>::Repr>, M>,
                <M as ConnectMapper<2>>::Err,
            >;

            fn mul(self, rhs: $r) -> Self::Output {
                let (lhs, [lhs_mapper]) = ToTensor::to_tensor(self).into_raw();
                let (rhs, [rhs_mapper]) = ToTensor::to_tensor(rhs).into_raw();
                ConnectMapper::<2>::connect([lhs_mapper, rhs_mapper]).map(
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
        }
    };
}

impl_mul!(Tensor<L, M>, Tensor<R, M>);
impl_mul!(&'l Tensor<L, M>, Tensor<R, M>, 'l);
impl_mul!(&'l mut Tensor<L, M>, Tensor<R, M>, 'l);
impl_mul!(Tensor<L, M>, &'r Tensor<R, M>, 'r);
impl_mul!(&'l Tensor<L, M>, &'r Tensor<R, M>, 'l, 'r);
impl_mul!(&'l mut Tensor<L, M>, &'r Tensor<R, M>, 'l, 'r);
impl_mul!(Tensor<L, M>, &'r mut Tensor<R, M>, 'r);
impl_mul!(&'l Tensor<L, M>, &'r mut Tensor<R, M>, 'l, 'r);
impl_mul!(&'l mut Tensor<L, M>, &'r mut Tensor<R, M>, 'l, 'r);

macro_rules! impl_mul_runtime {
    ($l:ty, $r:ty $(, $life:lifetime)*) => {
        impl<
            $($life,)*
            L: TensorTupleRepr<1>,
            R: TensorTupleRepr<1>,
            M: ConnectMapper<2> + Clone,
            RT: IsRuntime,
            Err,
        > Mul<$r> for $l
        where
            $l: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            $r: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<
                    MulRepr<
                        <$l as ToBoundTensorTuple<1>>::Repr,
                        <$r as ToBoundTensorTuple<1>>::Repr,
                    >,
                    M,
                >,
            >,
            <RT as RuntimeFor<
                Tensor<
                    MulRepr<
                        <$l as ToBoundTensorTuple<1>>::Repr,
                        <$r as ToBoundTensorTuple<1>>::Repr,
                    >,
                    M,
                >,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                MulRepr<
                    <$l as ToBoundTensorTuple<1>>::Repr,
                    <$r as ToBoundTensorTuple<1>>::Repr,
                >,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        MulRepr<
                            <$l as ToBoundTensorTuple<1>>::Repr,
                            <$r as ToBoundTensorTuple<1>>::Repr,
                        >,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<<M as ConnectMapper<2>>::Err, Err>,
            >;

            fn mul(self, rhs: $r) -> Self::Output {
                let (lhs, lhs_rt) = self.to_bound_tensor_tuple().into_raw();
                let (rhs, rhs_rt) = rhs.to_bound_tensor_tuple().into_raw();

                if lhs_rt != rhs_rt {
                    return Err(RuntimeErr::Runtime);
                }
                let rt = lhs_rt;
                let task = (lhs * rhs).map_err(RuntimeErr::Defer)?;
                let res = rt.ctx().execute(task).map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }
    };
}

impl_mul_runtime!(BoundTensor<L, M, RT>, BoundTensor<R, M, RT>);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, BoundTensor<R, M, RT>, 'l);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, BoundTensor<R, M, RT>, 'l);
impl_mul_runtime!(BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>, 'r);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>, 'l, 'r);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, &'r BoundTensor<R, M, RT>, 'l, 'r);
impl_mul_runtime!(BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>, 'r);
impl_mul_runtime!(&'l BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>, 'l, 'r);
impl_mul_runtime!(&'l mut BoundTensor<L, M, RT>, &'r mut BoundTensor<R, M, RT>, 'l, 'r);
