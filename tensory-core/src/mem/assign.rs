//! Assignment of a lazy tensor into an existing tensor representation.

use crate::{
    op::{BinaryEwiseRepr, EwiseExt, OverlayMapper},
    repr::TensorTupleRepr,
    tensor::{TensorTuple, ToTensorTuple},
};

pub struct AssignOp;

/// Lazy assignment tensor representation.
type AssignRepr<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> =
    BinaryEwiseRepr<N, S, D, AssignOp>;

/// Provides lazy assignment for tensor tuples with matching leg sets.
pub trait AssignExt<const N: usize>: ToTensorTuple<N> {
    /// Creates an assignment task from `self` into `destination`.
    fn assign<D>(
        self,
        dst: D,
    ) -> Result<
        TensorTuple<N, AssignRepr<N, Self::Repr, D::Repr>, Self::Mapper>,
        <Self::Mapper as OverlayMapper<2>>::Err,
    >
    where
        Self: Sized,
        D: ToTensorTuple<N, Mapper = Self::Mapper>,
        Self::Mapper: OverlayMapper<2>,
    {
        let src = self.to_tensor_tuple();
        let dst = dst.to_tensor_tuple();
        src.ewise2(dst, AssignOp)
    }
}
impl<const N: usize, T: ToTensorTuple<N>> AssignExt<N> for T {}
