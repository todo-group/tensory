#![no_std]
// #[cfg(test)]
// extern crate std;

pub mod arith;
pub mod linalg;
pub mod utils;

use core::marker::PhantomData;

use tensory_core::{
    concept::task::{Context, IsTask},
    repr::{AsViewRepr, TensorRepr, TensorTupleRepr},
    tensor::{Tensor, ToTensor},
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegulatedRepr<A: TensorTupleRepr<1>, C: CoefficientRepr<Scalar = N::Scalar>, N: Regulation> {
    repr: A,
    coeff: C,
    _reg: PhantomData<N>,
}

impl<A: TensorTupleRepr<1> + Clone, C: CoefficientRepr<Scalar = N::Scalar> + Clone, N: Regulation> Clone
    for RegulatedRepr<A, C, N>
{
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
            coeff: self.coeff.clone(),
            _reg: PhantomData,
        }
    }
}

/// Represents a coefficient carried alongside a regulated tensor.
///
/// # Safety
///
/// Implementations must preserve the coefficient algebra represented by each
/// operation and keep the associated scalar type consistent.
pub unsafe trait CoefficientRepr: Sized {
    type Scalar;
    fn build(scalar: Self::Scalar) -> Self;
    fn merge<const N: usize>(coeffies: [Self; N]) -> Self;
    fn mul<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self;
    fn div<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self;
    fn factorize<const N: usize>(coeffs: [Self; N]) -> (Self, [Self::Scalar; N]);
}

pub trait UnpackableCoefficientRepr: CoefficientRepr {
    fn unpack(self) -> Self::Scalar;
}

/// Defines the scalar type used by a regulation scheme.
///
/// # Safety
///
/// The associated scalar type must be valid for every regulator and inflator
/// implementation using this regulation scheme.
pub unsafe trait Regulation: Sized {
    type Scalar;
}
/// Normalizes a tensor representation and extracts its coefficient.
///
/// # Safety
///
/// Implementations must preserve the value represented by the input tensor and
/// return a coefficient compatible with the normalized representation.
pub unsafe trait Regulator<A: TensorTupleRepr<1>>: Regulation {
    type Res: TensorTupleRepr<1>;
    fn regulate(repr: A) -> (Self::Res, Self::Scalar);
    // fn scalar_mul(repr: A, scalar: Self::Scalar) -> A;
    // fn scalar_div(repr: A, scalar: Self::Scalar) -> A;
}
/// Normalizes a scalar and extracts its coefficient.
///
/// # Safety
///
/// Implementations must preserve the represented scalar value.
pub unsafe trait ScalarRegulator<E>: Regulation {
    fn scalar_regulate(scalar: E) -> (E, Self::Scalar);
}

// pub unsafe trait DivRegulator<A: TensorTupleRepr<1>, E>: Regulation {
//     type Res: TensorTupleRepr<1>;
//     fn div_regulate(repr: A, scalar: E) -> (Self::Res, Self::Scalar);
// }

/// Restores a regulated tensor using a coefficient.
///
/// # Safety
///
/// Implementations must preserve the axis structure and apply the coefficient
/// according to the regulation scheme.
pub unsafe trait Inflator<A: TensorTupleRepr<1>>: Regulation {
    type Res: TensorTupleRepr<1>;
    fn inflate(repr: A, coeff: Self::Scalar) -> Self::Res;
}

unsafe impl<A: TensorTupleRepr<1>, C: CoefficientRepr<Scalar = N::Scalar>, N: Regulation> TensorTupleRepr<1>
    for RegulatedRepr<A, C, N>
{
    fn naxes_array(&self) -> [usize; 1] {
        [self.repr.naxes()]
    }
}

impl<A: TensorTupleRepr<1>, C: CoefficientRepr<Scalar = N::Scalar>, N: Regulation> RegulatedRepr<A, C, N> {
    // pub fn new(repr: A, normalizer: N) -> Self {
    //     Self { repr, normalizer }
    // }
    pub fn into_raw(self) -> (A, C) {
        (self.repr, self.coeff)
    }
}

unsafe impl<
    'a,
    A: TensorTupleRepr<1> + AsViewRepr<'a, 1>,
    C: CoefficientRepr<Scalar = N::Scalar> + Clone,
    N: Regulation,
> AsViewRepr<'a, 1> for RegulatedRepr<A, C, N>
{
    type View = RegulatedRepr<A::View, C, N>;
    fn view(&'a self) -> Self::View {
        RegulatedRepr {
            repr: self.repr.view(),
            coeff: self.coeff.clone(),
            _reg: PhantomData,
        }
    }
}

// unsafe impl<
//     'a,
//     A: TensorTupleRepr<1> + AsViewMutRepr<'a>,
//     C: Coefficienty<Scalar = N::Scalar> + Clone,
//     N: RegulatorCore,
// > AsViewMutRepr<'a> for RegulatedRepr<A, C, N>
// {
//     type ViewMut = RegulatedRepr<A::ViewMut, C, N>;
//     fn view_mut(&'a mut self) -> Self::ViewMut {
//         RegulatedRepr {
//             repr: self.repr.view_mut(),
//             coeff: self.coeff.clone(),
//             _reg: PhantomData,
//         }
//     }
// }

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct RegulatedCtx<C>(C);

pub trait TensorRegulatedTask<C>: IsTask {
    fn with_regulated<Mk>(
        self,
        inner_ctx: C,
    ) -> <RegulatedCtx<C> as Context<Mk, Self>>::Output
    where
        Self: Sized,
        RegulatedCtx<C>: Context<Mk, Self>;
}

impl<T: IsTask, C> TensorRegulatedTask<C> for T {
    fn with_regulated<Mk>(
        self,
        inner_ctx: C,
    ) -> <RegulatedCtx<C> as Context<Mk, Self>>::Output
    where
        Self: Sized,
        RegulatedCtx<C>: Context<Mk, Self>,
    {
        RegulatedCtx(inner_ctx).execute(self)
    }
}

pub trait TensorDefaultRegulatedTask: IsTask {}

impl<T: IsTask> TensorDefaultRegulatedTask for T {}

pub trait RegulatedTensorExt<A: TensorTupleRepr<1>, C: CoefficientRepr<Scalar = N::Scalar>, N: Regulation>:
    ToTensor<Repr = RegulatedRepr<A, C, N>> + Sized
{
    fn into_inner_tensor(self) -> (Tensor<A, Self::Mapper>, C);
    fn into_normal_tensor(self) -> Tensor<N::Res, Self::Mapper>
    where
        C: UnpackableCoefficientRepr,
        N: Inflator<A>;
}
impl<
    A: TensorTupleRepr<1>,
    C: CoefficientRepr<Scalar = N::Scalar>,
    N: Regulation,
    T: ToTensor<Repr = RegulatedRepr<A, C, N>>,
> RegulatedTensorExt<A, C, N> for T
{
    fn into_inner_tensor(self) -> (Tensor<A, Self::Mapper>, C) {
        let (norm_repr, mapper) = self.to_tensor().into_raw();
        (
            unsafe { Tensor::from_raw_unchecked(norm_repr.repr, mapper) },
            norm_repr.coeff,
        )
    }
    fn into_normal_tensor(self) -> Tensor<N::Res, Self::Mapper>
    where
        C: UnpackableCoefficientRepr,
        N: Inflator<A>,
    {
        let (norm_repr, mapper) = self.to_tensor().into_raw();
        let raw_repr = norm_repr.repr;
        let coeff = norm_repr.coeff;
        let repr = N::inflate(raw_repr, coeff.unpack());

        unsafe { Tensor::from_raw_unchecked(repr, mapper) }
    }
}

pub trait ToRegulatedTensorExt: ToTensor + Sized {
    fn regulate<C: CoefficientRepr<Scalar = N::Scalar>, N: Regulator<Self::Repr>>(
        self,
    ) -> Tensor<RegulatedRepr<N::Res, C, N>, Self::Mapper>;
}

impl<T: ToTensor> ToRegulatedTensorExt for T {
    fn regulate<C: CoefficientRepr<Scalar = N::Scalar>, N: Regulator<Self::Repr>>(
        self,
    ) -> Tensor<RegulatedRepr<N::Res, C, N>, Self::Mapper> {
        let (repr, mapper) = self.to_tensor().into_raw();
        let (repr, coeff) = N::regulate(repr);
        let coeff = C::build(coeff);
        unsafe {
            Tensor::from_raw_unchecked(
                RegulatedRepr {
                    repr,
                    coeff,
                    _reg: PhantomData,
                },
                mapper,
            )
        }
    }
}

// add ,sub, などにも実装
// 通常のTensorと共通化できるように
// normをとってlog, また符号や位相 <= normalizerなくても使いたい

// -scaled? chatgptに聞いてみる
// scalarが交換することは要請として共通化
// 正規化は要請しない

// ([a] * A) * ([b] * B) = [a*b] * (A*B = X) = [a*b*x'] * X'
// ([a] * A) + ([b] * B) = [x] * (a/x * A + b/x * B = Y) = [x*y'] * Y'
// ([a] * A) - ([b] * B) = [x] * (a/x * A - b/x * B = Y) = [x*y'] * Y'
// ([a] * A) * x = [a*r] * A * p
// ([a] * A) / x = [a/r] * A / p
// [x] * X = [x] * (U * S * V) = [x] * (u * U') * (s * S') * (v * V') = (x * u * s * v) * (U' * S' * V') ... A -> (A', [a]) no! (A [1]) -> (A', [a])

// by regulation
// t_reg A -> (A', a)
// c_reg x -> (r, p )

// by coefficienty
// merge [[a],[b],[c]...] -> [a*b*c...]
// scalar_merge [x], a,b,c,... -> [x*a*b*c...]
// factorize [[a],[b],[c]...] -> [x], a/x, b/x, c/x ...
// mul [a], x -> [a*x]
// div [a], x -> [a/x]

// ([a] * A) * ([b] * B) = [a*b] * (A*B = X) = [a*b*x'] * X'
// ... merge, [mul], t_reg, scalar_merge
// ([a] * A) + ([b] * B) = [x] * (a/x * A + b/x * B = Y) = [x*y'] * Y'
// ([a] * A) - ([b] * B) = [x] * (a/x * A - b/x * B = Y) = [x*y'] * Y'
// ... factorize, [scalar_mul], [scalar_mul], [add/sub], t_reg. scalar_merge
// ([a] * A) * x = [a*r] * A * p
// ([a] * A) / x = [a/r] * A / p
// ... c_reg, mul/div, [scalar_mul/sclar_div]

// [x] * X = [x] * (U * S * V) = [x] * (u * U') * (s * S') * (v * V') = [x * u * s * v] * (U' * S' * V') ... A -> (A', [a]) no! (A [1]) -> (A', [a])
// ... [svd], t_reg, t_reg, t_reg, scalar_merge

// found V,S s.t. ([x] * X) V = V S
// <=> found V' S' s.t. X V' = V' S' then ([x] X) V' = V' ([x] S') so V = V', S = [x] S'

// ([x] * X)^k = [x^k] * X^k
//... pow

// note!!!
// ([a] * A) + ([b] * B) = [x] * (a/x * A + b/x * B = Y) = [x*y'] * Y'
// ([a] * A) - ([b] * B) = [x] * (a/x * A - b/x * B = Y) = [x*y'] * Y'
// only they require complex operation for A and B:
// this is problem: we declare &A + &B only for avoid

//require [scalar_mul/sclar_div] avail with () ctx
// this bound enables mock imple for regualtor: calc coeff without changing repr, then div repr by it

// unsafe traitの要請にこれを入れる
