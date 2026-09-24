use core::ops::Add;

use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, RuntimeErr, RuntimeFor},
    },
    op::OverlayMapper,
    op::{BinaryEwiseRepr, EwiseExt},
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

/// Operation for element-wise addition.
pub struct AddOp;

/// Lazy representation for an addition operation using the `AddOp`.
pub type AddRepr<const N: usize, L, R> = BinaryEwiseRepr<N, L, R, AddOp>;

// 9 combinations of Lhs/Rhs being owned/view/view_mut

macro_rules! impl_add {
    ($l:ty,$r:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* L: TensorTupleRepr<1>, R: TensorTupleRepr<1>, M: OverlayMapper<2>> Add<$r> for $l
        where
            $l: ToTensor<Mapper = M>,
            $r: ToTensor<Mapper = M>,
        {
            type Output = Result<
                Tensor<AddRepr<1, <$l as ToTensor>::Repr, <$r as ToTensor>::Repr>, M>,
                <M as OverlayMapper<2>>::Err,
            >;
            fn add(self, rhs: $r) -> Self::Output {
                let lhs = ToTensor::to_tensor(self);
                let rhs = ToTensor::to_tensor(rhs);
                lhs.ewise2(rhs, AddOp)
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
            RT: RuntimeFor<Tensor<AddRepr<1,<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>,
            <RT as RuntimeFor<Tensor<AddRepr<1,<$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                AddRepr<1, <$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M,
                CType = Resulting<Raw,Err>
            >
        {
            type Output = Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<RT::Mk, 1, AddRepr<1, <$l as ToBoundTensorTuple<1>>::Repr, <$r as ToBoundTensorTuple<1>>::Repr>, M>>::Repr,
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
                let rt = lhs_rt;
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
