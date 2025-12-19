#![no_std]
#[cfg(test)]
extern crate std;

use tensory_core::{
    arith::{MulCtxImpl, TensorMul},
    mapper::{AxisMapper, ConnectAxisOrigin},
    repr::{AsViewMutRepr, AsViewRepr, TensorRepr},
    tensor::{Tensor, TensorTask, ToTensor},
};
use tensory_linalg::norm;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct NormalizedRepr<R: TensorRepr, N: Normalizer<R>> {
    repr: R,
    normalizer: N,
}

pub unsafe trait Normalizer<R: TensorRepr>: Sized {}

pub trait CanonicalizeNormalizer<R: TensorRepr>: Normalizer<R> {
    fn normalize(original_repr: R) -> Result<(Self, R), Self::Err>;
    type Err;
}

pub trait MulNormalizer<L: TensorRepr, R: TensorRepr, C: MulCtxImpl<L, R>>:
    Normalizer<L> + Normalizer<R> + Normalizer<C::Res>
{
    type Err;
    fn mul(
        lhs_ner: Self,
        lhs_repr: L,
        rhs_ner: Self,
        rhs_repr: R,
        ctx: C,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<(Self, C::Res), Self::Err>;
}

unsafe impl<R: TensorRepr, N: Normalizer<R>> TensorRepr for NormalizedRepr<R, N> {
    fn naxes(&self) -> usize {
        self.repr.naxes()
    }
}

impl<R: TensorRepr, N: Normalizer<R>> NormalizedRepr<R, N> {
    // pub fn new(repr: R, normalizer: N) -> Self {
    //     Self { repr, normalizer }
    // }
    pub fn into_raw(self) -> (R, N) {
        (self.repr, self.normalizer)
    }
}

unsafe impl<'a, R: TensorRepr + AsViewRepr<'a>, N: Normalizer<R> + Normalizer<R::View> + Clone>
    AsViewRepr<'a> for NormalizedRepr<R, N>
{
    type View = NormalizedRepr<R::View, N>;
    fn view(&'a self) -> Self::View {
        NormalizedRepr {
            repr: self.repr.view(),
            normalizer: self.normalizer.clone(),
        }
    }
}

unsafe impl<
    'a,
    R: TensorRepr + AsViewMutRepr<'a>,
    N: Normalizer<R> + Normalizer<R::ViewMut> + Clone,
> AsViewMutRepr<'a> for NormalizedRepr<R, N>
{
    type ViewMut = NormalizedRepr<R::ViewMut, N>;
    fn view_mut(&'a mut self) -> Self::ViewMut {
        NormalizedRepr {
            repr: self.repr.view_mut(),
            normalizer: self.normalizer.clone(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct NormalizedCtx<C>(C);

pub trait TensorNormalizedTask<C>: TensorTask<NormalizedCtx<C>> {
    fn with_normalized(self, inner_ctx: C) -> Self::Output;
}

impl<T: TensorTask<NormalizedCtx<C>>, C> TensorNormalizedTask<C> for T {
    fn with_normalized(self, inner_ctx: C) -> Self::Output {
        self.with(NormalizedCtx(inner_ctx))
    }
}

unsafe impl<
    C: MulCtxImpl<Lhs, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Normalizer<Lhs> + Normalizer<Rhs> + MulNormalizer<Lhs, Rhs, C>,
> MulCtxImpl<NormalizedRepr<Lhs, N>, NormalizedRepr<Rhs, N>> for NormalizedCtx<C>
{
    type Res = NormalizedRepr<C::Res, N>;
    type Err = N::Err;

    unsafe fn mul_unchecked(
        self,
        lhs: NormalizedRepr<Lhs, N>,
        rhs: NormalizedRepr<Rhs, N>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (res_ner, res) = N::mul(
            lhs.normalizer,
            lhs.repr,
            rhs.normalizer,
            rhs.repr,
            self.0,
            axis_origin,
        )?;
        Ok(NormalizedRepr {
            repr: res,
            normalizer: res_ner,
        })
    }
}

pub trait NormalizedTensorExt<A: TensorRepr, N: Normalizer<A>>:
    ToTensor<Repr = NormalizedRepr<A, N>> + Sized
{
    fn into_inner_tensor(self) -> (Tensor<A, Self::Mapper>, N);
}
impl<A: TensorRepr, N: Normalizer<A>, T: ToTensor<Repr = NormalizedRepr<A, N>>>
    NormalizedTensorExt<A, N> for T
{
    fn into_inner_tensor(self) -> (Tensor<A, Self::Mapper>, N) {
        let (norm_repr, mapper) = self.to_tensor().into_raw();
        (
            unsafe { Tensor::from_raw_unchecked(norm_repr.repr, mapper) },
            norm_repr.normalizer,
        )
    }
}

pub trait ToNormalizedTensorExt: ToTensor + Sized {
    fn normalize<N: Normalizer<Self::Repr> + CanonicalizeNormalizer<Self::Repr>>(
        self,
    ) -> Result<Tensor<NormalizedRepr<Self::Repr, N>, Self::Mapper>, N::Err>;
}

impl<T: ToTensor> ToNormalizedTensorExt for T {
    fn normalize<N: Normalizer<Self::Repr> + CanonicalizeNormalizer<Self::Repr>>(
        self,
    ) -> Result<Tensor<NormalizedRepr<Self::Repr, N>, Self::Mapper>, N::Err> {
        let (repr, mapper) = self.to_tensor().into_raw();
        let (normalizer, repr) = N::normalize(repr)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NormalizedRepr { repr, normalizer }, mapper) })
    }
}

#[cfg(test)]
mod tests {
    use core::{any, ops::Mul};
    use std::{ops::Div, println};

    use ndarray_linalg::{Lapack, Norm, Scalar, norm};
    use num_traits::ConstZero;
    use tensory_basic::mapper::VecMapper;
    use tensory_core::{
        arith::{CommutativeScalarDivCtx, MulCtx, MulCtxImpl},
        prelude::*,
    };
    use tensory_linalg::norm::{NormCtx, TensorNormExt};
    use tensory_ndarray::{
        NdDenseRepr, NdDenseTensor, NdDenseTensorExt, NdDenseViewRepr, TenalgError,
    };

    use crate::{
        CanonicalizeNormalizer, MulNormalizer, NormalizedTensorExt, Normalizer,
        TensorNormalizedTask, ToNormalizedTensorExt,
    };

    type Tensor<E, I> = NdDenseTensor<E, VecMapper<I>>;

    #[derive(Clone, Copy)]
    struct L2Normalizer<E>(E);

    unsafe impl<E: Scalar> Normalizer<NdDenseRepr<E>> for L2Normalizer<E::Real> {}
    unsafe impl<'a, E: Scalar> Normalizer<NdDenseViewRepr<'a, E>> for L2Normalizer<E::Real> {}

    impl<E: Scalar> CanonicalizeNormalizer<NdDenseRepr<E>> for L2Normalizer<E::Real>
    where
        E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
    {
        type Err = TenalgError;
        fn normalize(repr: NdDenseRepr<E>) -> Result<(Self, NdDenseRepr<E>), Self::Err> {
            let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
            let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
                (),
                repr,
                E::from_real(norm),
            )?;
            Ok((Self(norm), repr))
        }
    }

    impl<
        'l,
        'r,
        E: Scalar,
        //C: MulCtxImpl<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, Res = NdDenseRepr<E>>,
    > MulNormalizer<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, ()> for L2Normalizer<E::Real>
    where
        E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
    {
        type Err = TenalgError;
        fn mul(
            lhs_ner: Self,
            lhs_repr: NdDenseViewRepr<'l, E>,
            rhs_ner: Self,
            rhs_repr: NdDenseViewRepr<'r, E>,
            ctx: (),
            axis_origin: ConnectAxisOrigin<2>,
        ) -> Result<(Self, NdDenseRepr<E>), TenalgError> {
            let repr = ctx.mul(lhs_repr, rhs_repr, axis_origin)?;
            let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
            let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
                (),
                repr,
                E::from_real(norm),
            )?;
            let norm = lhs_ner.0 * rhs_ner.0;
            Ok((Self(norm), repr))
        }
    }

    #[test]
    fn it_works() -> anyhow::Result<()> {
        let a = Tensor::<f64, _>::random(lm!["a"=>2,"b"=>3]).unwrap();
        let b = Tensor::<f64, _>::random(lm!["b"=>3,"c"=>4]).unwrap();

        let a_ned = a.clone().normalize::<L2Normalizer<_>>().unwrap();
        let b_ned = b.clone().normalize::<_>().unwrap();
        let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

        //println!("{:?}", (&a_unned).norm().exec().unwrap());

        let a_mul_b = (&a * &b)?.with(())?;
        let a_ned_mul_b_ned = (&a_ned * &b_ned)?.with_normalized(())?;
        let a_mul_b_ned = a_mul_b.normalize::<L2Normalizer<_>>().unwrap();

        let (a_ned_mul_b_ned_unned, a_ned_mul_b_ned_norm) = a_ned_mul_b_ned.into_inner_tensor();
        let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

        let diff = (&a_ned_mul_b_ned_unned - &a_mul_b_ned_unned)?.with(())?;

        println!("diff_norm: {}", (&diff).norm().exec()?);
        println!(
            "norm_diff: {}",
            (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
        );

        Ok(())
    }
}


// add ,sub, などにも実装
// 通常のTensorと共通化できるように
// normをとってlog, また符号や位相 <= normalizerなくても使いたい

