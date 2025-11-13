use tensory_core::{
    arith::{MulCtxImpl, TensorMul},
    mapper::{AxisMapper, ConnectAxisOrigin},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};
use tensory_linalg::norm;

pub struct NormalizedRepr<R: TensorRepr, N: Normalizer<R>> {
    repr: R,
    normalizer: N,
}

trait Normalizer<R: TensorRepr>: Sized {
    fn normalize(self, original_repr: R) -> (Self, R);
}
trait MulNormalizer<R: TensorRepr>: Normalizer<R> {
    fn mul(lhs_norm: Self, rhs_norm: Self, res: R) -> (R, Self);
}

unsafe impl<R: TensorRepr, N: Normalizer<R>> TensorRepr for NormalizedRepr<R, N> {
    fn naxes(&self) -> usize {
        self.repr.naxes()
    }
}

impl<R: TensorRepr, N: Normalizer<R>> NormalizedRepr<R, N> {
    pub fn new(repr: R, normalizer: N) -> Self {
        Self { repr, normalizer }
    }
}

pub struct NormalizedCtx<C>(C);

pub trait TensorNormalizedTask<C>: TensorTask<NormalizedCtx<C>> {
    fn with_normalized(self, inner_ctx: C) -> Self::Output;
}

impl<T: TensorTask<NormalizedCtx<C>>, C> TensorNormalizedTask<C> for T {
    fn with_normalized(self, inner_ctx: C) -> Self::Output {
        self.with(NormalizedCtx(inner_ctx))
    }
}

//a*b .with_normalized(ctx);

unsafe impl<
    C: MulCtxImpl<Lhs, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Normalizer<Lhs> + Normalizer<Rhs> + MulNormalizer<C::Res>,
> MulCtxImpl<NormalizedRepr<Lhs, N>, NormalizedRepr<Rhs, N>> for NormalizedCtx<C>
{
    type Res = NormalizedRepr<C::Res, N>;
    type Err = C::Err;

    unsafe fn mul_unchecked(
        self,
        lhs: NormalizedRepr<Lhs, N>,
        rhs: NormalizedRepr<Rhs, N>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        let res = unsafe { self.0.mul_unchecked(lhs.repr, rhs.repr, axis_origin) }?;
        let (res, normalizer) = N::mul(lhs.normalizer, rhs.normalizer, res);
        Ok(NormalizedRepr {
            repr: res,
            normalizer,
        })
        //todo!()
    }
}

trait ToNormalizedTensorExt: ToTensor + Sized {
    fn normalize<N: Normalizer<Self::Repr>>(
        self,
        normalizer: N,
    ) -> Tensor<NormalizedRepr<Self::Repr, N>, Self::Mapper>;
}

impl<T: ToTensor> ToNormalizedTensorExt for T {
    fn normalize<N: Normalizer<Self::Repr>>(
        self,
        normalizer: N,
    ) -> Tensor<NormalizedRepr<Self::Repr, N>, Self::Mapper> {
        let (repr, mapper) = self.to_tensor().into_raw();
        let repr = normalizer.normalize(repr);
        unsafe { Tensor::from_raw_unchecked(NormalizedRepr { repr, normalizer }, mapper) }
    }
}

#[cfg(test)]
mod tests {
    use ndarray_linalg::Scalar;
    use tensory_basic::mapper::VecMapper;
    use tensory_core::prelude::*;
    use tensory_ndarray::{NdDenseRepr, NdDenseTensor, NdDenseTensorExt};

    use crate::{Normalizer, ToNormalizedTensorExt};

    type Tensor<E, I> = NdDenseTensor<E, VecMapper<I>>;

    struct L2Normalizer<E>(E);
    impl<E: Scalar> Normalizer<NdDenseRepr<E>> for L2Normalizer<E::Real> {
        fn normalize(&self, original_repr: NdDenseRepr<E>) -> NdDenseRepr<E> {
            todo!()
        }
    }

    #[test]
    fn it_works() {
        let a = Tensor::<f64, _>::random(lm!["a"=>2,"b"=>3]).unwrap();
        let b = Tensor::<f64, _>::random(lm!["b"=>3,"c"=>4]).unwrap();

        let a = a.normalize(normalizer);
    }
}
