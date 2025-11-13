use tensory_core::{mapper::EquivGroupedAxes, repr::TensorRepr};

/// Raw context of diagonalization operation: A= V*D*VC.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait ExpCtxImpl<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs power operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn exp_unchecked(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err>;
}
