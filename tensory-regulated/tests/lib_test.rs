use core::marker::PhantomData;
use std::println;

use ndarray::ScalarOperand;
use ndarray::Zip;
use ndarray_linalg::c64;
use ndarray_linalg::{Lapack, Scalar};
use tensory_basic::mapper::VecMapper;
use tensory_core::{
    arith::{CommutativeScalarDivCtx, CommutativeScalarMulCtx},
    prelude::*,
};
use tensory_linalg::{
    norm::{NormCtx, TensorNormExt},
    svd::TensorSvdExt,
};
use tensory_ndarray::regulated::L2Regulator;
use tensory_ndarray::regulated::LnCoeff;
use tensory_ndarray::{NdDenseRepr, NdDenseTensor, NdDenseTensorExt, NdDenseViewRepr, TenalgErr};

use tensory_regulated::Inflator;
//use crate::arith::WeightedAddCtxImplReordered;
use tensory_regulated::{
    CoefficientRepr, RegulatedTensorExt, Regulation, Regulator, TensorDefaultRegulatedTask,
    TensorRegulatedTask, ToRegulatedTensorExt, UnpackableCoefficientRepr,
    arith::WeightedAddCtxImpl,
};

type Tensor<E, I> = NdDenseTensor<E, VecMapper<I>>;

//unsafe impl<E: Scalar> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {}
//unsafe impl<'a, E: Scalar> Regulator<NdDenseViewRepr<'a, E>> for L2Regulator<E::Real> {}

// impl<E: Scalar> CanonicalizeNormalizer<NdDenseRepr<E>> for L2Regulator<E::Real>
// where
//     E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
// {
//     type Err = TenalgErr;
//     fn normalize(repr: NdDenseRepr<E>) -> Result<(Self, NdDenseRepr<E>), Self::Err> {
//         let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
//         let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
//             (),
//             repr,
//             E::from_real(norm),
//         )?;
//         Ok((Self(norm), repr))
//     }
// }

// #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Error)]
// enum NormalizerErr<EO, EN> {
//     #[error("operation error: {0}")]
//     Op(EO),
//     #[error("normalization error")]
//     Normalize(EN),
// }

// struct ConsumeAddCtx<C>(C);
// unsafe impl<
//     Lhs: TensorRepr + for<'a> AsViewRepr<'a>,
//     Rhs: TensorRepr + for<'a> AsViewRepr<'a>,
//     C: for<'a, 'b> AddCtxImpl<
//             <Lhs as AsViewRepr<'a>>::View,
//             <Rhs as AsViewRepr<'b>>::View,
//             Res = Res,
//             Err = Err,
//         >,
//     Res: TensorRepr,
//     Err,
// > AddCtxImpl<Lhs, Rhs> for ConsumeAddCtx<C>
// {
//     type Res = Res;

//     type Err = Err;

//     unsafe fn add_unchecked(
//         self,
//         lhs: Lhs,
//         rhs: Rhs,
//         axis_mapping: OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         unsafe { C::add_unchecked(self.0, lhs.view(), rhs.view(), axis_mapping) }
//     }
// }

// unsafe impl<
//     'l,
//     'r,
//     E: Scalar,
//     C: MulCtxImpl<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, Res = NdDenseRepr<E>>,
// > MulNormalizer<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, C> for L2Regulator<E::Real>
// where
//     E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
// {
//     type Err = NormalizerErr<C::Err, TenalgErr>;
//     fn mul(
//         lhs_ner: Self,
//         lhs_repr: NdDenseViewRepr<'l, E>,
//         rhs_ner: Self,
//         rhs_repr: NdDenseViewRepr<'r, E>,
//         ctx: C,
//         axis_origin: ConnectAxisOrigin<2>,
//     ) -> Result<(Self, NdDenseRepr<E>), Self::Err> {
//         let repr = ctx
//             .mul(lhs_repr, rhs_repr, axis_origin)
//             .map_err(NormalizerErr::Op)?;
//         let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
//         let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
//             (),
//             repr,
//             E::from_real(norm),
//         )
//         .map_err(NormalizerErr::Normalize)?;
//         let norm = lhs_ner.0 * rhs_ner.0;
//         Ok((Self(norm), repr))
//     }
// }

// unsafe impl<E: Scalar, C: SvdCtxImpl<NdDenseRepr<E>>> SvdNormalizer<NdDenseRepr<E>, C>
//     for L2Regulator<E::Real>
// {
//     type Err = TenalgErr;
//     fn svd(
//         a_ner: Self,
//         a_repr: NdDenseRepr<E>,
//         ctx: C,
//         axes_split: GroupedAxes<2>,
//     ) -> Result<
//         (
//             Self,
//             NdDenseRepr<E>,
//             Self,
//             NdDenseRepr<E>,
//             Self,
//             NdDenseRepr<E>,
//         ),
//         Self::Err,
//     > {
//         let (u_repr, s_repr, v_repr) = ctx.svd(a_repr, axes_split)?;
//         let (u_ner, u_repr) = <L2Regulator<E::Real> as CanonicalizeNormalizer<
//             NdDenseRepr<E>,
//         >>::normalize(u_repr)?;
//         let (v_ner, v_repr) = <L2Regulator<E::Real> as CanonicalizeNormalizer<
//             NdDenseRepr<E>,
//         >>::normalize(v_repr)?;
//         Ok((
//             u_ner,
//             u_repr,
//             Self(s_repr.norm().unwrap()),
//             s_repr,
//             v_ner,
//             v_repr,
//         ))
//     }
// }

#[test]
fn it_works() -> anyhow::Result<()> {
    let a = Tensor::<c64, _>::random(lm!["a"=>20,"b"=>30]).unwrap();
    let b = Tensor::<c64, _>::random(lm!["b"=>30,"c"=>40]).unwrap();

    let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();
    let b_red = b.clone().regulate();
    //let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

    //println!("{:?}", (&a_unned).norm().exec().unwrap());
    let a_mul_b = (&a * &b)?.exec()?;
    let a_red_mul_b_red = (&a_red * &b_red)?.exec()?;
    //let a_mul_b_red = a_mul_b.regulate::<LnCoeff<f64>, L2Regulator<_>>();

    let a_red_mul_b_red_unred = a_red_mul_b_red.into_normal_tensor();

    //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

    let a_mul_b_diff = (&a_red_mul_b_red_unred - &a_mul_b)?.with(())?;

    println!(
        "diff_norm: {}",
        (&a_mul_b_diff).norm().exec()? / (&a_mul_b).norm().exec()?
    );
    // println!(
    //     "norm_diff: {}",
    //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
    // );

    Ok(())
}

// pub struct Weighted;

// unsafe impl<'l, 'r, E: Scalar>
//     WeightedAddCtxImpl<E::Real, NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>>
//     for Weighted
// {
//     type Res = NdDenseRepr<E>;
//     type Err = TenalgErr;

//     unsafe fn weighted_add_unchecked(
//         self,
//         lhs_coeff: E::Real,
//         lhs: NdDenseViewRepr<'l, E>,
//         rhs_coeff: E::Real,
//         rhs: NdDenseViewRepr<'r, E>,
//         axis_mapping: tensory_core::mapper::OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         let lhs_raw = lhs.into_raw();
//         let rhs_raw = rhs.into_raw();

//         let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

//         let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
//         let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

//         if lhs_raw.dim() == rhs_raw.dim() {
//             Ok(NdDenseRepr::from_raw(
//                 Zip::from(lhs_raw)
//                     .and(rhs_raw)
//                     .map_collect(|l, r| l.mul_real(lhs_coeff) + r.mul_real(rhs_coeff)),
//             ))
//         } else {
//             Err(TenalgErr::InvalidInput)
//         }
//     }
// }

#[test]
fn it_works_add() -> anyhow::Result<()> {
    let a = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();
    let b = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();

    let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();
    let b_red = b.clone().regulate();
    //let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

    //println!("{:?}", (&a_unned).norm().exec().unwrap());
    let a_add_b = (&a + &b)?.exec()?;
    let a_red_add_b_red = (&a_red + &b_red)?.exec()?;
    //<() as WeightedAddCtxImpl<f64, NdDenseViewRepr<f64>, f64, NdDenseViewRepr<f64>>>::weighted_add_unchecked;

    //let a_mul_b_red = a_mul_b.regulate::<LnCoeff<f64>, L2Regulator<_>>();

    let a_red_add_b_red_unred = a_red_add_b_red.into_normal_tensor();

    //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

    let diff = (&a_red_add_b_red_unred - &a_add_b)?.with(())?;
    println!("diff_norm: {}", (&diff).norm().exec()?);
    // println!(
    //     "norm_diff: {}",
    //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
    // );

    Ok(())
}

#[test]
fn it_works_svd() -> anyhow::Result<()> {
    let a = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();

    let (u, s, v) = (&a).svd(ls![&"a", &"b"], "us", "sv")?.exec()?;

    let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();

    let (u_red, s_red, v_red) = (&a_red).svd(ls![&"a", &"b"], "us", "sv")?.exec()?;

    let u_red_unred = u_red.into_normal_tensor();
    let s_red_unred = s_red.into_normal_tensor();
    let v_red_unred = v_red.into_normal_tensor();

    //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

    let u_diff = (&u_red_unred - &u)?.with(())?;
    let s_diff = (&s_red_unred - &s)?.with(())?;
    let v_diff = (&v_red_unred - &v)?.with(())?;
    println!(
        "u_diff_norm: {}",
        (&u_diff).norm().exec()? / (&u).norm().exec()?
    );
    println!(
        "s_diff_norm: {}",
        (&s_diff).norm().exec()? / (&s).norm().exec()?
    );
    println!(
        "v_diff_norm: {}",
        (&v_diff).norm().exec()? / (&v).norm().exec()?
    );

    // println!(
    //     "norm_diff: {}",
    //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
    // );

    Ok(())
}

// #[cfg(test)]
// mod tests2 {
//     use super::Coefficienty;
//     use super::UnpackCoefficienty;
//     use super::arith::WeightedAddCtxImpl;
//     use core::marker::PhantomData;
//     use ndarray_linalg::Lapack;
//     use ndarray_linalg::Scalar;
//     use std::println;
//     use tensory_basic::mapper::VecMapper;
//     use tensory_ndarray::NdDenseTensor;

//     use super::Regulator;
//     use super::RegulatorCore;
//     use ndarray::ScalarOperand;
//     use ndarray::Zip;
//     use tensory_core::{
//         arith::{CommutativeScalarDivCtx, CommutativeScalarMulCtx},
//         prelude::*,
//     };
//     use tensory_linalg::{
//         norm::{NormCtx, TensorNormExt},
//         svd::TensorSvdExt,
//     };
//     use tensory_ndarray::{NdDenseRepr, NdDenseTensorExt, NdDenseViewRepr, TenalgErr};

//     type Tensor<E, I> = NdDenseTensor<E, VecMapper<I>>;

//     struct L2Regulator<E>(PhantomData<E>);

//     unsafe impl<E> RegulatorCore for L2Regulator<E> {
//         type Scalar = E;
//     }
//     unsafe impl<E: Lapack + Scalar + ScalarOperand> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {
//         fn regulate(repr: NdDenseRepr<E>) -> (NdDenseRepr<E>, Self::Scalar) {
//             let norm = <() as NormCtx<_>>::norm((), repr.view()).unwrap(); // Infallible
//             let repr =
//                 <() as CommutativeScalarDivCtx<_, _>>::scalar_div((), repr, E::from_real(norm))
//                     .unwrap();
//             (repr, norm)
//         }
//         fn scalar_mul(repr: NdDenseRepr<E>, scalar: Self::Scalar) -> NdDenseRepr<E> {
//             <() as CommutativeScalarMulCtx<_, _>>::scalar_mul((), repr, E::from_real(scalar))
//                 .unwrap()
//         }
//         fn scalar_div(repr: NdDenseRepr<E>, scalar: Self::Scalar) -> NdDenseRepr<E> {
//             <() as CommutativeScalarDivCtx<_, _>>::scalar_div((), repr, E::from_real(scalar))
//                 .unwrap()
//         }
//     }

//     #[derive(Clone, Copy)]
//     struct LnCoeff<E: Scalar>(E::Real);

//     unsafe impl<E: Scalar> Coefficienty for LnCoeff<E> {
//         type Scalar = E::Real;

//         fn build(scalar: Self::Scalar) -> Self {
//             Self(scalar.ln())
//         }

//         fn merge<const N: usize>(coeffies: [Self; N]) -> Self {
//             let ln_sum = coeffies.iter().map(|c| c.0).sum();
//             Self(ln_sum)
//         }

//         fn mul<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
//             let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc + c.ln());
//             Self(ln_sum)
//         }

//         fn div<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
//             let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc - c.ln());
//             Self(ln_sum)
//         }

//         fn factorize<const N: usize>(coeffs: [Self; N]) -> (Self, [Self::Scalar; N]) {
//             let ln_max = coeffs
//                 .iter()
//                 .map(|c| c.0)
//                 .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
//                 .unwrap();
//             (Self(ln_max), coeffs.map(|coeff| (coeff.0 - ln_max).exp()))
//         }
//     }
//     impl<E: Scalar> UnpackCoefficienty for LnCoeff<E> {
//         fn unpack(self) -> Self::Scalar {
//             self.0.exp()
//         }
//     }

//     use super::ToRegulatedTensorExt;
//     #[test]
//     fn main() -> anyhow::Result<()> {
//         let a = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();
//         let b = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();

//         let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();
//         let b_red = b.clone().regulate();
//         //let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

//         //println!("{:?}", (&a_unned).norm().exec().unwrap());
//         let a_add_b = (&a + &b)?.exec()?;
//         let a_red_add_b_red =
//             (&a_red + &b_red)?.with_regulated(tensory_ndarray::regulated::Weighted)?;
//         //let a_mul_b_red = a_mul_b.regulate::<LnCoeff<f64>, L2Regulator<_>>();

//         let a_red_add_b_red_unred = a_red_add_b_red.into_normal_tensor();

//         //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

//         let diff = (&a_red_add_b_red_unred - &a_add_b)?.with(())?;
//         println!("diff_norm: {}", (&diff).norm().exec()?);
//         // println!(
//         //     "norm_diff: {}",
//         //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
//         // );

//         Ok(())
//     }

//     struct Weighted;
//     use super::RegulatedTensorExt;
//     use super::TensorRegulatedTask;

//     // unsafe impl<'l, 'r, E: Scalar>
//     //     WeightedAddCtxImpl<E::Real, NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>>
//     //     for Weighted
//     // {
//     //     type Res = NdDenseRepr<E>;
//     //     type Err = TenalgErr;

//     //     unsafe fn weighted_add_unchecked(
//     //         self,
//     //         lhs_coeff: E::Real,
//     //         lhs: NdDenseViewRepr<'l, E>,
//     //         rhs_coeff: E::Real,
//     //         rhs: NdDenseViewRepr<'r, E>,
//     //         axis_mapping: tensory_core::mapper::OverlayAxisMapping<2>,
//     //     ) -> Result<Self::Res, Self::Err> {
//     //         let lhs_raw = lhs.into_raw();
//     //         let rhs_raw = rhs.into_raw();

//     //         let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

//     //         let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
//     //         let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

//     //         if lhs_raw.dim() == rhs_raw.dim() {
//     //             Ok(NdDenseRepr::from_raw(
//     //                 Zip::from(lhs_raw)
//     //                     .and(rhs_raw)
//     //                     .map_collect(|l, r| l.mul_real(lhs_coeff) + r.mul_real(rhs_coeff)),
//     //             ))
//     //         } else {
//     //             Err(TenalgErr::InvalidInput)
//     //         }
//     //     }
//     // }
// }

// should move to tests
// #[cfg(test)]
// mod tests {
//     use core::marker::PhantomData;
//     use std::println;

//     use ndarray::ScalarOperand;
//     use ndarray::Zip;
//     use ndarray_linalg::c64;
//     use ndarray_linalg::{Lapack, Scalar};
//     use tensory_basic::mapper::VecMapper;
//     use tensory_core::{
//         arith::{CommutativeScalarDivCtx, CommutativeScalarMulCtx},
//         prelude::*,
//     };
//     use tensory_linalg::{
//         norm::{NormCtx, TensorNormExt},
//         svd::TensorSvdExt,
//     };
//     use tensory_ndarray::{
//         NdDenseRepr, NdDenseTensor, NdDenseTensorExt, NdDenseViewRepr, TenalgErr,
//     };

//     //use crate::arith::WeightedAddCtxImplReordered;
//     use crate::{
//         Coefficienty, RegulatedTensorExt, Regulator, RegulatorCore, TensorDefaultRegulatedTask,
//         TensorRegulatedTask, ToRegulatedTensorExt, UnpackCoefficienty, arith::WeightedAddCtxImpl,
//     };

//     type Tensor<E, I> = NdDenseTensor<E, VecMapper<I>>;

//     //#[derive(Clone, Copy)]
//     struct L2Regulator<E>(PhantomData<E>);

//     unsafe impl<E> RegulatorCore for L2Regulator<E> {
//         type Scalar = E;
//     }
//     unsafe impl<E: Lapack + Scalar + ScalarOperand> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {
//         fn regulate(repr: NdDenseRepr<E>) -> (NdDenseRepr<E>, Self::Scalar) {
//             let norm = <() as NormCtx<_>>::norm((), repr.view()).unwrap(); // Infallible
//             let repr =
//                 <() as CommutativeScalarDivCtx<_, _>>::scalar_div((), repr, E::from_real(norm))
//                     .unwrap();
//             (repr, norm)
//         }
//         fn scalar_mul(repr: NdDenseRepr<E>, scalar: Self::Scalar) -> NdDenseRepr<E> {
//             <() as CommutativeScalarMulCtx<_, _>>::scalar_mul((), repr, E::from_real(scalar))
//                 .unwrap()
//         }
//         fn scalar_div(repr: NdDenseRepr<E>, scalar: Self::Scalar) -> NdDenseRepr<E> {
//             <() as CommutativeScalarDivCtx<_, _>>::scalar_div((), repr, E::from_real(scalar))
//                 .unwrap()
//         }
//     }

//     #[derive(Clone, Copy)]
//     struct LnCoeff<E: Scalar>(E::Real);

//     unsafe impl<E: Scalar> Coefficienty for LnCoeff<E> {
//         type Scalar = E::Real;

//         fn build(scalar: Self::Scalar) -> Self {
//             Self(scalar.ln())
//         }

//         fn merge<const N: usize>(coeffies: [Self; N]) -> Self {
//             let ln_sum = coeffies.iter().map(|c| c.0).sum();
//             Self(ln_sum)
//         }

//         fn mul<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
//             let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc + c.ln());
//             Self(ln_sum)
//         }

//         fn div<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
//             let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc - c.ln());
//             Self(ln_sum)
//         }

//         fn factorize<const N: usize>(coeffs: [Self; N]) -> (Self, [Self::Scalar; N]) {
//             let ln_max = coeffs
//                 .iter()
//                 .map(|c| c.0)
//                 .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
//                 .unwrap();
//             (Self(ln_max), coeffs.map(|coeff| (coeff.0 - ln_max).exp()))
//         }
//     }
//     impl<E: Scalar> UnpackCoefficienty for LnCoeff<E> {
//         fn unpack(self) -> Self::Scalar {
//             self.0.exp()
//         }
//     }

//     //unsafe impl<E: Scalar> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {}
//     //unsafe impl<'a, E: Scalar> Regulator<NdDenseViewRepr<'a, E>> for L2Regulator<E::Real> {}

//     // impl<E: Scalar> CanonicalizeNormalizer<NdDenseRepr<E>> for L2Regulator<E::Real>
//     // where
//     //     E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
//     // {
//     //     type Err = TenalgErr;
//     //     fn normalize(repr: NdDenseRepr<E>) -> Result<(Self, NdDenseRepr<E>), Self::Err> {
//     //         let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
//     //         let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
//     //             (),
//     //             repr,
//     //             E::from_real(norm),
//     //         )?;
//     //         Ok((Self(norm), repr))
//     //     }
//     // }

//     // #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Error)]
//     // enum NormalizerErr<EO, EN> {
//     //     #[error("operation error: {0}")]
//     //     Op(EO),
//     //     #[error("normalization error")]
//     //     Normalize(EN),
//     // }

//     // struct ConsumeAddCtx<C>(C);
//     // unsafe impl<
//     //     Lhs: TensorRepr + for<'a> AsViewRepr<'a>,
//     //     Rhs: TensorRepr + for<'a> AsViewRepr<'a>,
//     //     C: for<'a, 'b> AddCtxImpl<
//     //             <Lhs as AsViewRepr<'a>>::View,
//     //             <Rhs as AsViewRepr<'b>>::View,
//     //             Res = Res,
//     //             Err = Err,
//     //         >,
//     //     Res: TensorRepr,
//     //     Err,
//     // > AddCtxImpl<Lhs, Rhs> for ConsumeAddCtx<C>
//     // {
//     //     type Res = Res;

//     //     type Err = Err;

//     //     unsafe fn add_unchecked(
//     //         self,
//     //         lhs: Lhs,
//     //         rhs: Rhs,
//     //         axis_mapping: OverlayAxisMapping<2>,
//     //     ) -> Result<Self::Res, Self::Err> {
//     //         unsafe { C::add_unchecked(self.0, lhs.view(), rhs.view(), axis_mapping) }
//     //     }
//     // }

//     // unsafe impl<
//     //     'l,
//     //     'r,
//     //     E: Scalar,
//     //     C: MulCtxImpl<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, Res = NdDenseRepr<E>>,
//     // > MulNormalizer<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>, C> for L2Regulator<E::Real>
//     // where
//     //     E: Lapack + ndarray::ScalarOperand + Div<Output = E>,
//     // {
//     //     type Err = NormalizerErr<C::Err, TenalgErr>;
//     //     fn mul(
//     //         lhs_ner: Self,
//     //         lhs_repr: NdDenseViewRepr<'l, E>,
//     //         rhs_ner: Self,
//     //         rhs_repr: NdDenseViewRepr<'r, E>,
//     //         ctx: C,
//     //         axis_origin: ConnectAxisOrigin<2>,
//     //     ) -> Result<(Self, NdDenseRepr<E>), Self::Err> {
//     //         let repr = ctx
//     //             .mul(lhs_repr, rhs_repr, axis_origin)
//     //             .map_err(NormalizerErr::Op)?;
//     //         let norm = <() as NormCtx<NdDenseViewRepr<E>>>::norm((), repr.view()).unwrap();
//     //         let repr = <() as CommutativeScalarDivCtx<NdDenseRepr<E>, E>>::scalar_div(
//     //             (),
//     //             repr,
//     //             E::from_real(norm),
//     //         )
//     //         .map_err(NormalizerErr::Normalize)?;
//     //         let norm = lhs_ner.0 * rhs_ner.0;
//     //         Ok((Self(norm), repr))
//     //     }
//     // }

//     // unsafe impl<E: Scalar, C: SvdCtxImpl<NdDenseRepr<E>>> SvdNormalizer<NdDenseRepr<E>, C>
//     //     for L2Regulator<E::Real>
//     // {
//     //     type Err = TenalgErr;
//     //     fn svd(
//     //         a_ner: Self,
//     //         a_repr: NdDenseRepr<E>,
//     //         ctx: C,
//     //         axes_split: GroupedAxes<2>,
//     //     ) -> Result<
//     //         (
//     //             Self,
//     //             NdDenseRepr<E>,
//     //             Self,
//     //             NdDenseRepr<E>,
//     //             Self,
//     //             NdDenseRepr<E>,
//     //         ),
//     //         Self::Err,
//     //     > {
//     //         let (u_repr, s_repr, v_repr) = ctx.svd(a_repr, axes_split)?;
//     //         let (u_ner, u_repr) = <L2Regulator<E::Real> as CanonicalizeNormalizer<
//     //             NdDenseRepr<E>,
//     //         >>::normalize(u_repr)?;
//     //         let (v_ner, v_repr) = <L2Regulator<E::Real> as CanonicalizeNormalizer<
//     //             NdDenseRepr<E>,
//     //         >>::normalize(v_repr)?;
//     //         Ok((
//     //             u_ner,
//     //             u_repr,
//     //             Self(s_repr.norm().unwrap()),
//     //             s_repr,
//     //             v_ner,
//     //             v_repr,
//     //         ))
//     //     }
//     // }

//     #[test]
//     fn it_works() -> anyhow::Result<()> {
//         let a = Tensor::<c64, _>::random(lm!["a"=>20,"b"=>30]).unwrap();
//         let b = Tensor::<c64, _>::random(lm!["b"=>30,"c"=>40]).unwrap();

//         let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();
//         let b_red = b.clone().regulate();
//         //let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

//         //println!("{:?}", (&a_unned).norm().exec().unwrap());
//         let a_mul_b = (&a * &b)?.exec()?;
//         let a_red_mul_b_red = (&a_red * &b_red)?.exec_regulated()?;
//         //let a_mul_b_red = a_mul_b.regulate::<LnCoeff<f64>, L2Regulator<_>>();

//         let a_red_mul_b_red_unred = a_red_mul_b_red.into_normal_tensor();

//         //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

//         let a_mul_b_diff = (&a_red_mul_b_red_unred - &a_mul_b)?.with(())?;

//         println!(
//             "diff_norm: {}",
//             (&a_mul_b_diff).norm().exec()? / (&a_mul_b).norm().exec()?
//         );
//         // println!(
//         //     "norm_diff: {}",
//         //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
//         // );

//         Ok(())
//     }

//     // pub struct Weighted;

//     // unsafe impl<'l, 'r, E: Scalar>
//     //     WeightedAddCtxImpl<E::Real, NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>>
//     //     for Weighted
//     // {
//     //     type Res = NdDenseRepr<E>;
//     //     type Err = TenalgErr;

//     //     unsafe fn weighted_add_unchecked(
//     //         self,
//     //         lhs_coeff: E::Real,
//     //         lhs: NdDenseViewRepr<'l, E>,
//     //         rhs_coeff: E::Real,
//     //         rhs: NdDenseViewRepr<'r, E>,
//     //         axis_mapping: tensory_core::mapper::OverlayAxisMapping<2>,
//     //     ) -> Result<Self::Res, Self::Err> {
//     //         let lhs_raw = lhs.into_raw();
//     //         let rhs_raw = rhs.into_raw();

//     //         let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

//     //         let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
//     //         let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

//     //         if lhs_raw.dim() == rhs_raw.dim() {
//     //             Ok(NdDenseRepr::from_raw(
//     //                 Zip::from(lhs_raw)
//     //                     .and(rhs_raw)
//     //                     .map_collect(|l, r| l.mul_real(lhs_coeff) + r.mul_real(rhs_coeff)),
//     //             ))
//     //         } else {
//     //             Err(TenalgErr::InvalidInput)
//     //         }
//     //     }
//     // }

//     #[test]
//     fn it_works_add() -> anyhow::Result<()> {
//         let a = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();
//         let b = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();

//         let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();
//         let b_red = b.clone().regulate();
//         //let (a_unned, a_norm) = a_ned.clone().into_inner_tensor();

//         //println!("{:?}", (&a_unned).norm().exec().unwrap());
//         let a_add_b = (&a + &b)?.exec()?;
//         let a_red_add_b_red = (&a_red + &b_red)?.with_regulated(())?;
//         //<() as WeightedAddCtxImpl<f64, NdDenseViewRepr<f64>, f64, NdDenseViewRepr<f64>>>::weighted_add_unchecked;

//         //let a_mul_b_red = a_mul_b.regulate::<LnCoeff<f64>, L2Regulator<_>>();

//         let a_red_add_b_red_unred = a_red_add_b_red.into_normal_tensor();

//         //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

//         let diff = (&a_red_add_b_red_unred - &a_add_b)?.with(())?;
//         println!("diff_norm: {}", (&diff).norm().exec()?);
//         // println!(
//         //     "norm_diff: {}",
//         //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
//         // );

//         Ok(())
//     }

//     #[test]
//     fn it_works_svd() -> anyhow::Result<()> {
//         let a = Tensor::<f64, _>::random(lm!["a"=>20,"b"=>30,"c"=>40]).unwrap();

//         let (u, s, v) = (&a).svd(ls![&"a", &"b"], "us", "sv")?.exec()?;

//         let a_red = a.clone().regulate::<LnCoeff<f64>, L2Regulator<_>>();

//         let (u_red, s_red, v_red) = (&a_red)
//             .svd(ls![&"a", &"b"], "us", "sv")?
//             .exec_regulated()?;

//         let u_red_unred = u_red.into_normal_tensor();
//         let s_red_unred = s_red.into_normal_tensor();
//         let v_red_unred = v_red.into_normal_tensor();

//         //let (a_mul_b_ned_unned, a_mul_b_ned_norm) = a_mul_b_ned.into_inner_tensor();

//         let u_diff = (&u_red_unred - &u)?.with(())?;
//         let s_diff = (&s_red_unred - &s)?.with(())?;
//         let v_diff = (&v_red_unred - &v)?.with(())?;
//         println!(
//             "u_diff_norm: {}",
//             (&u_diff).norm().exec()? / (&u).norm().exec()?
//         );
//         println!(
//             "s_diff_norm: {}",
//             (&s_diff).norm().exec()? / (&s).norm().exec()?
//         );
//         println!(
//             "v_diff_norm: {}",
//             (&v_diff).norm().exec()? / (&v).norm().exec()?
//         );

//         // println!(
//         //     "norm_diff: {}",
//         //     (a_ned_mul_b_ned_norm.0 - a_mul_b_ned_norm.0).abs()
//         // );

//         Ok(())
//     }
// }
