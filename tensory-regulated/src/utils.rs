use tensory_core::utils::axis_info::AxisInfoReprImpl;

use crate::{CoefficientRepr, RegulatedRepr, Regulation};

impl<'a, A, C: CoefficientRepr<Scalar = N::Scalar>, N: Regulation> AxisInfoReprImpl<'a, 1>
    for RegulatedRepr<A, C, N>
where
    A: AxisInfoReprImpl<'a, 1>,
{
    type AxisInfo = A::AxisInfo;

    unsafe fn axis_info_unchecked(&'a self, i: usize, j: usize) -> Self::AxisInfo {
        unsafe { self.repr.axis_info_unchecked(i, j) }
    }
}
