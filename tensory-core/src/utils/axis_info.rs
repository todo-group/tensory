use crate::{
    mapper::{AxisMapper, TranslateMapper},
    repr::{TensorRepr, TensorTupleRepr},
    tensor::{Tensor, TensorTuple},
};

/// Tensor representation providing the information of each axis, WITHOUT checking bounds.
pub trait AxisInfoReprImpl<'a, const N: usize>: TensorTupleRepr<N> {
    /// The type of the axis information.
    type AxisInfo;
    /// Returns the information for the given axis, WITHOUT checking bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i < N, j < self.naxes()`.
    unsafe fn axis_info_unchecked(&'a self, i: usize, j: usize) -> Self::AxisInfo;
}

/// Safe version of `AxisInfoImpl`.
///
/// The blanket implementation checks bounds.
pub trait AxisInfoRepr<'a, const N: usize>: AxisInfoReprImpl<'a, N> {
    /// Returns the information for the given axis, with checking bounds.
    fn axis_info(&'a self, i: usize, j: usize) -> Self::AxisInfo;
}
impl<'a, const N: usize, T: AxisInfoReprImpl<'a, N>> AxisInfoRepr<'a, N> for T {
    fn axis_info(&'a self, i: usize, j: usize) -> Self::AxisInfo {
        if i >= N || j >= self.naxes_array()[i] {
            panic!("dim not match")
        }
        unsafe { self.axis_info_unchecked(i, j) }
    }
}

impl<
    'a,
    'i,
    const N: usize,
    A: AxisInfoReprImpl<'a, N>,
    Id,
    M: AxisMapper<Id = Id> + TranslateMapper<&'i Id, Res = usize>,
> TensorTuple<N, A, M>
where
    Id: 'i,
{
    /// Returns axis information selected by its mapper ID.
    pub fn axis_info(&'a self, i: usize, leg: &'i M::Id) -> Result<A::AxisInfo, M::Err> {
        if i >= N {
            panic!("repr not match")
        }
        let j = self.mapper_array()[i].translate(leg)?;
        // if j >= self.repr().naxes_array()[i] {
        //     panic!("dim not match")
        // }
        unsafe { Ok(self.repr().axis_info_unchecked(i, j)) }
    }
}
