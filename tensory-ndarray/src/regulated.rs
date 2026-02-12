use core::marker::PhantomData;

use ndarray::{ScalarOperand, Zip};
use ndarray_linalg::{Lapack, Scalar};
use tensory_core::{
    arith::{CommutativeScalarDivCtx, CommutativeScalarMulCtx},
    repr::AsViewRepr,
};
use tensory_linalg::norm::NormCtx;
use tensory_regulated::{
    CoefficientRepr, Inflator, Regulation, Regulator, UnpackableCoefficientRepr,
    arith::WeightedAddCtxImpl,
};

use crate::{NdDenseRepr, NdDenseViewRepr, TenalgErr};

pub struct L2Regulator<E>(PhantomData<E>);

unsafe impl<E> Regulation for L2Regulator<E> {
    type Scalar = E;
}
unsafe impl<E: Lapack + ScalarOperand> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {
    type Res = NdDenseRepr<E>;

    fn regulate(repr: NdDenseRepr<E>) -> (NdDenseRepr<E>, Self::Scalar) {
        let norm = ().norm(repr.view()).unwrap(); // Infallible
        let repr = <() as CommutativeScalarDivCtx<_, _>>::scalar_div((), repr, E::from_real(norm))
            .unwrap();
        (repr, norm)
    }
}
unsafe impl<E: Lapack + ScalarOperand> Inflator<NdDenseRepr<E>> for L2Regulator<E::Real> {
    type Res = NdDenseRepr<E>;

    fn inflate(repr: NdDenseRepr<E>, coeff: Self::Scalar) -> Self::Res {
        <() as CommutativeScalarMulCtx<_, _>>::scalar_mul((), repr, E::from_real(coeff)).unwrap()
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct LnCoeff<E: Scalar>(E::Real);
impl<E: Scalar> LnCoeff<E> {
    pub fn ln(self) -> E::Real {
        self.0
    }
}

unsafe impl<E: Scalar> CoefficientRepr for LnCoeff<E> {
    type Scalar = E::Real;

    fn build(scalar: Self::Scalar) -> Self {
        Self(scalar.ln())
    }

    fn merge<const N: usize>(coeffies: [Self; N]) -> Self {
        let ln_sum = coeffies.iter().map(|c| c.0).sum();
        Self(ln_sum)
    }

    fn mul<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
        let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc + c.ln());
        Self(ln_sum)
    }

    fn div<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
        let ln_sum = coeffs.iter().fold(self.0, |acc, &c| acc - c.ln());
        Self(ln_sum)
    }

    fn factorize<const N: usize>(coeffs: [Self; N]) -> (Self, [Self::Scalar; N]) {
        let ln_max = coeffs
            .iter()
            .map(|c| c.0)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap();
        (Self(ln_max), coeffs.map(|coeff| (coeff.0 - ln_max).exp()))
    }
}
impl<E: Scalar> UnpackableCoefficientRepr for LnCoeff<E> {
    fn unpack(self) -> Self::Scalar {
        self.0.exp()
    }
}

unsafe impl<'l, 'r, E: Scalar>
    WeightedAddCtxImpl<E::Real, NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgErr;

    unsafe fn weighted_add_unchecked(
        self,
        lhs_coeff: E::Real,
        lhs: NdDenseViewRepr<'l, E>,
        rhs_coeff: E::Real,
        rhs: NdDenseViewRepr<'r, E>,
        axis_mapping: tensory_core::mapper::OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let lhs_raw = lhs.into_raw();
        let rhs_raw = rhs.into_raw();

        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

        let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
        let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

        if lhs_raw.dim() == rhs_raw.dim() {
            Ok(NdDenseRepr::from_raw(
                Zip::from(lhs_raw)
                    .and(rhs_raw)
                    .map_collect(|l, r| l.mul_real(lhs_coeff) + r.mul_real(rhs_coeff)),
            ))
        } else {
            Err(TenalgErr::InvalidInput)
        }
    }
}

// unsafe impl<'l, 'r, E: Scalar>
//     WeightedAddCtxImplReordered<NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>, E::Real>
//     for ()
// {
//     type Res = NdDenseRepr<E>;
//     type Err = TenalgErr;

//     unsafe fn weighted_add_unchecked_r(
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
