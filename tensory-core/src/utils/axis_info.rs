use crate::{
    mapper::{AxisMapper, SortMapper, TranslateMapper},
    repr::TensorRepr,
    tensor::Tensor,
};

/// Tensor representation providing the information of each axis, WITHOUT checking bounds.
pub trait AxisInfoReprImpl<'a>: TensorRepr {
    /// The type of the axis information.
    type AxisInfo;
    /// Returns the information for the given axis, WITHOUT checking bounds.
    unsafe fn axis_info_unchecked(&'a self, i: usize) -> Self::AxisInfo;
}

/// Safe version of `AxisInfoImpl`.
///
/// The blanket implementation checks bounds.
pub trait AxisInfoRepr<'a>: AxisInfoReprImpl<'a> {
    /// Returns the information for the given axis, with checking bounds.
    fn axis_info(&'a self, i: usize) -> Self::AxisInfo;
}
impl<'a, T: AxisInfoReprImpl<'a>> AxisInfoRepr<'a> for T {
    fn axis_info(&'a self, i: usize) -> Self::AxisInfo {
        if i >= self.naxes() {
            panic!("dim not match")
        }
        unsafe { self.axis_info_unchecked(i) }
    }
}

impl<
    'a,
    'i,
    A: AxisInfoReprImpl<'a>,
    Id,
    M: AxisMapper<Id = Id> + TranslateMapper<&'i Id, Res = usize>,
> Tensor<A, M>
where
    Id: 'i,
{
    pub fn axis_info(&'a self, leg: &'i M::Id) -> Result<A::AxisInfo, M::Err> {
        let i = self.mapper().translate(leg)?;
        if i >= self.repr().naxes() {
            panic!("dim not match")
        }
        unsafe { Ok(self.repr().axis_info_unchecked(i)) }
    }
}
