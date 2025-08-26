use crate::core::tensor::TensorRepr;

/// Tensor representation providing the information of each axis, WITHOUT checking bounds.
pub trait AxisInfoImpl: TensorRepr {
    /// The type of the axis information.
    type AxisInfo;
    /// Returns the information for the given axis, WITHOUT checking bounds.
    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo;
}

/// Safe version of `AxisInfoImpl`.
///
/// The blanket implementation checks bounds.
pub trait AxisInfo: AxisInfoImpl {
    /// Returns the information for the given axis, with checking bounds.
    fn axis_info(&self, i: usize) -> Result<Self::AxisInfo, AxisOutOfRangeError>;
}
impl<T: AxisInfoImpl> AxisInfo for T {
    fn axis_info(&self, i: usize) -> Result<Self::AxisInfo, AxisOutOfRangeError> {
        if i < self.dim() {
            Ok(unsafe { self.axis_info_unchecked(i) })
        } else {
            Err(AxisOutOfRangeError)
        }
    }
}
