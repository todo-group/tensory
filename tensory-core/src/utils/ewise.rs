use crate::{mapper::OverlayAxisMapping, repr::TensorRepr};

/// Raw context of element-wise operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the proper "axis structure" inherited from the input tensors described with `axis_mapping`.
pub unsafe trait EwiseCtxImpl<Lhs: TensorRepr, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
    ///
    /// # Safety
    ///
    /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
    unsafe fn ewise_unchecked(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}

/// Safe version if `EwiseCtxImpl`.
///
/// The blanket implementation checks input and panic if the condition is not satisfied.
pub trait EwiseCtx<Lhs: TensorRepr, Rhs: TensorRepr>: EwiseCtxImpl<Lhs, Rhs> {
    /// Safe version of `ewise_unchecked`.
    fn ewise(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}
impl<C: EwiseCtxImpl<Lhs, Rhs>, Lhs: TensorRepr, Rhs: TensorRepr> EwiseCtx<Lhs, Rhs> for C {
    fn ewise(
        self,
        lhs: Lhs,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let n_l = lhs.naxes();
        let n_r = rhs.naxes();
        let n = axis_mapping.naxes();
        if n_l != n || n_r != n {
            panic!("axis_mapping must match the number of axes with lhs and rhs");
        }
        unsafe { self.ewise_unchecked(lhs, rhs, axis_mapping) }
    }
}
