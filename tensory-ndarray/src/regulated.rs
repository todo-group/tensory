use core::marker::PhantomData;

use ndarray::{ScalarOperand, Zip};
use ndarray_linalg::{Lapack, Scalar};
use tensory_core::{
    arith::{
        LeftScalarDivOp, LeftScalarDivRepr, LeftScalarMulOp, LeftScalarMulRepr,
        RightScalarDivOp, RightScalarDivRepr, RightScalarMulOp, RightScalarMulRepr,
    },
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::Context,
    },
    op::AxisAllocation,
    repr::{AsViewRepr, TensorRepr},
};
use tensory_linalg::norm::NormRepr;
use tensory_regulated::{
    arith::{WeightedAddCtxImpl, WeightedSubCtxImpl},
    CoefficientRepr, Inflator, Regulation, Regulator, UnpackableCoefficientRepr,
};

use crate::{NdDenseRepr, NdDenseViewRepr, TenalgErr};

pub struct L2Regulator<E>(PhantomData<E>);

unsafe impl<E> Regulation for L2Regulator<E> {
    type Scalar = E;
}

unsafe impl<E: Lapack + ScalarOperand> Regulator<NdDenseRepr<E>> for L2Regulator<E::Real> {
    type Res = NdDenseRepr<E>;

    fn regulate(repr: NdDenseRepr<E>) -> (Self::Res, Self::Scalar) {
        let norm = ().execute(NormRepr::from_raw(repr.view())).unwrap();
        let scalar = E::from_real(norm);
        let axis_mapping = AxisAllocation::id(repr.naxes());
        let task = unsafe {
            RightScalarDivRepr::from_raw_unchecked((
                repr,
                [axis_mapping],
                RightScalarDivOp(scalar),
            ))
        };
        let repr = ().execute(task).unwrap();
        (repr, norm)
    }
}

unsafe impl<E: Lapack + ScalarOperand> Inflator<NdDenseRepr<E>> for L2Regulator<E::Real> {
    type Res = NdDenseRepr<E>;

    fn inflate(repr: NdDenseRepr<E>, coeff: Self::Scalar) -> Self::Res {
        let axis_mapping = AxisAllocation::id(repr.naxes());
        let task = unsafe {
            RightScalarMulRepr::from_raw_unchecked((
                repr,
                [axis_mapping],
                RightScalarMulOp(E::from_real(coeff)),
            ))
        };
        ().execute(task).unwrap()
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
        Self(coeffies.iter().map(|c| c.0).sum())
    }

    fn mul<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
        Self(coeffs.iter().fold(self.0, |acc, &c| acc + c.ln()))
    }

    fn div<const N: usize>(self, coeffs: [Self::Scalar; N]) -> Self {
        Self(coeffs.iter().fold(self.0, |acc, &c| acc - c.ln()))
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
        axis_mapping: AxisAllocation<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();
        let lhs = lhs.into_raw().permuted_axes(lhs_perm);
        let rhs = rhs.into_raw().permuted_axes(rhs_perm);
        if lhs.dim() != rhs.dim() {
            return Err(TenalgErr::InvalidInput);
        }
        Ok(NdDenseRepr::from_raw(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs.mul_real(lhs_coeff) + rhs.mul_real(rhs_coeff)),
        ))
    }
}

unsafe impl<'l, 'r, E: Scalar>
    WeightedSubCtxImpl<E::Real, NdDenseViewRepr<'l, E>, E::Real, NdDenseViewRepr<'r, E>> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgErr;

    unsafe fn weighted_sub_unchecked(
        self,
        lhs_coeff: E::Real,
        lhs: NdDenseViewRepr<'l, E>,
        rhs_coeff: E::Real,
        rhs: NdDenseViewRepr<'r, E>,
        axis_mapping: AxisAllocation<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();
        let lhs = lhs.into_raw().permuted_axes(lhs_perm);
        let rhs = rhs.into_raw().permuted_axes(rhs_perm);
        if lhs.dim() != rhs.dim() {
            return Err(TenalgErr::InvalidInput);
        }
        Ok(NdDenseRepr::from_raw(
            Zip::from(lhs)
                .and(rhs)
                .map_collect(|lhs, rhs| lhs.mul_real(lhs_coeff) - rhs.mul_real(rhs_coeff)),
        ))
    }
}