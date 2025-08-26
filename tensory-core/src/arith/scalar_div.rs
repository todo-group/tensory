use crate::tensor::TensorRepr;

/// Raw context of left scalar division operation.
///
/// This trait is unsafe because the implementation must ensure that the result tensor must have the same axis structure as the input tensor.
pub unsafe trait LeftScalarDivContext<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs left scalar division operation on the tensor `a`.
    ///
    /// # Safety
    ///
    /// the implementor must ensure the result tensor has the same axis structure as the input tensor.
    fn left_scalar_div(self, a: A, scalar: E) -> Result<Self::Res, Self::Err>;
}
