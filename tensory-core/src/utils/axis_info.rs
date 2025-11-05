use crate::{mapper::TranslateMapper, repr::TensorRepr, tensor::Tensor};

/// Tensor representation providing the information of each axis, WITHOUT checking bounds.
pub trait AxisInfoReprImpl: TensorRepr {
    /// The type of the axis information.
    type AxisInfo;
    /// Returns the information for the given axis, WITHOUT checking bounds.
    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo;
}

/// Safe version of `AxisInfoImpl`.
///
/// The blanket implementation checks bounds.
pub trait AxisInfoRepr: AxisInfoReprImpl {
    /// Returns the information for the given axis, with checking bounds.
    fn axis_info(&self, i: usize) -> Self::AxisInfo;
}
impl<T: AxisInfoReprImpl> AxisInfoRepr for T {
    fn axis_info(&self, i: usize) -> Self::AxisInfo {
        if i >= self.naxes() {
            panic!("dim not match")
        }
        unsafe { self.axis_info_unchecked(i) }
    }
}

// impl<A: AxisInfoReprImpl, M: TranslateMapper> Tensor<A, M> {
//     fn axis_info(&self, leg: M::Id) -> Self::AxisInfo {
//         if i >= self.naxes() {
//             panic!("dim not match")
//         }
//         unsafe { self.axis_info_unchecked(i) }
//     }
// }
