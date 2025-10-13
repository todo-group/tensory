//! Layer 1 tensor concept: tensor with axes 0-indexed with usize, only valid in the tensor.

/// Minimal interface for tensor representations.
///
/// In the conceptual model, a tensor representation is a structured data object with multiple axes, each indexed from `0` up to `dim() - 1`.
///
/// In practice, a type implementing this trait serves as a handle for resource allocation and management.
///
/// # Safety
///
/// The implementor MUST ensure the following invariants:
///
/// - The number of axes of the tensor representation is fixed for the same object, even through mutable operations.
/// - The "semantic order" of the axes are never changed for the same object, even through mutable operations.
///
/// We refer the above invariants as "semantic structure of axes", or simply "axis structure". Here, the "semantic order" refers to the assignment of the usize indices to the conceptual axes, which are fully distinguished. See `Tensor` for more integrated explanation of "semantic order" (described as "semantic assignment").
///
/// `mem::{swap,replace,take,...}` syntactically violate the above conditons, but these operations semantically do not change the objects but move them. So we think the above conditions are not violated by these operations.
pub unsafe trait TensorRepr: Sized {
    /// Returns the number of axes of the tensor. this number is fixed for the same object even through mutable operations.
    ///
    /// this function serves as a dynamic version of `const N:usize`.
    fn dim(&self) -> usize;
}

/// Interface to generate a immutable view representation of itself.
///
/// # Safety
///
/// The implementor MUST ensure that the view representation has the same semantic structure of axes as the original representation.
pub unsafe trait AsViewRepr<'a>: TensorRepr {
    /// Immutable view representation type.
    type View: TensorRepr;
    /// Returns an immutable view representation of itself. The view has the same semantic structure of axes as the original representation.
    fn view(&'a self) -> Self::View;
}

/// Interface to generate a mutable view representation of itself.
///
/// # Safety
///
/// The implementor MUST ensure that the view representation has the same semantic structure of axes as the original representation.
pub unsafe trait AsViewMutRepr<'a>: TensorRepr {
    /// Mutable view representation type.
    type ViewMut: TensorRepr;
    /// Returns a mutable view representation of itself. The view has the same semantic structure of axes as the original representation.
    fn view_mut(&'a mut self) -> Self::ViewMut;
}
