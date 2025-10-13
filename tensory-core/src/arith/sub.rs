use crate::{mapper::OverlayAxisMapping, repr::TensorRepr};

/// Raw context of left subtraction operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor has same number of axes as the input tensors.
pub unsafe trait LeftSubCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left subtraction operation on the tensors `lhs` and `rhs` with the given axis pairs, and returns the result tensor each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that `axis_origin` is for the number of axes same as the input tensors.
    ///
    /// the implementor must ensure the result tensor has same number of axes as the input tensors.
    unsafe fn left_sub_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_origin: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
