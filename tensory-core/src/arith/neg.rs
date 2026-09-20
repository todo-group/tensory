use crate::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor},
    },
    mapper::AxisMapper,
    repr::TensorTupleRepr,
    tensor::{BoundTensor, Tensor, TensorTupleContext, ToBoundTensorTuple, ToTensor},
};

use core::{convert::Infallible, ops::Neg};

/*
/// Raw context of negation operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the same "axis structure" as the input tensor.
pub unsafe trait NegCtx<A: TensorTupleRepr<1>> {
    /// The type of the result tensor representation.
    type Res: TensorTupleRepr<1>;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs negation operation on the tensor `a`.
    fn negate(self, a: A) -> Result<Self::Res, Self::Err>;
}

*/

/// Lazy representation for a negation operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct NegRepr<A: TensorTupleRepr<1>> {
    a: A,
}

impl<A: TensorTupleRepr<1>> NegRepr<A> {
    /// Creates a representation for an input tensor.
    pub fn from_raw(a: A) -> Self {
        Self { a }
    }

    /// Creates a representation without checking its input.
    ///
    /// # Safety
    ///
    /// `a` must be a valid tensor representation.
    pub unsafe fn from_raw_unchecked(a: A) -> Self {
        Self { a }
    }

    /// Decomposes the representation into its input.
    pub fn into_raw(self) -> A {
        self.a
    }
}

unsafe impl<A: TensorTupleRepr<1>> TensorTupleRepr<1> for NegRepr<A> {
    fn naxes_array(&self) -> [usize; 1] {
        self.a.naxes_array()
    }
}

impl<A: TensorTupleRepr<1>> IsTask for NegRepr<A> {}

macro_rules! impl_neg {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper> Neg for $a
        where
            $a: ToTensor<Mapper = M>,
        {
            type Output = Tensor<NegRepr<<$a as ToTensor>::Repr>, M>;
            fn neg(self) -> Self::Output {
                let (a, [mapper]) = ToTensor::to_tensor(self).into_raw();
                unsafe { Tensor::from_raw_unchecked(NegRepr::from_raw(a), [mapper]) }
            }
        }
    };
}

impl_neg!(Tensor<A, M>);
impl_neg!(&'a Tensor<A, M>,'a);
impl_neg!(&'a mut Tensor<A, M>,'a);

// Runtime-bound implementations for negation.
macro_rules! impl_neg_runtime {
    ($a:ty $(,$life:lifetime)* ) => {
        impl<$($life,)* A: TensorTupleRepr<1>, M: AxisMapper + Clone, RT: IsRuntime, Err> Neg for $a
        where
            $a: ToBoundTensorTuple<1, Mapper = M, Runtime = RT>,
            RT: RuntimeFor<
                Tensor<NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>,
            >,
            <RT as RuntimeFor<
                Tensor<NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>, M>,
            >>::Ctx: TensorTupleContext<
                RT::Mk,
                1,
                NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>,
                M,
                CType = Resulting<Raw, Err>,
            >,
        {
            type Output =
            Result<
                BoundTensor<
                    <RT::Ctx as TensorTupleContext<
                        RT::Mk,
                        1,
                        NegRepr<<$a as ToBoundTensorTuple<1>>::Repr>,
                        M,
                    >>::Repr,
                    M,
                    RT,
                >,
                RuntimeErr<Infallible, Err>,
            >;
            fn neg(self) -> Self::Output {
                let (a, rt) = self.to_bound_tensor_tuple().into_raw();
                let res = rt
                    .ctx()
                    .execute(-a)
                    .map_err(RuntimeErr::Execute)?;
                Ok(BoundTensor::from_raw(res, rt))
            }
        }
    };
}

impl_neg_runtime!(BoundTensor<A, M,RT>);
impl_neg_runtime!(&'a BoundTensor<A, M,RT>,'a);
impl_neg_runtime!(&'a mut BoundTensor<A, M,RT>,'a);
