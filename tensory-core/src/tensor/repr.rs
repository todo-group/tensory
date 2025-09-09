/// A minimal interface for tensor representations.
///
/// This trait requires
/// In mental model, tensor representation is a structured data with multiple dimensions (axes).
///
/// In practice, a tensor representation is treated as a handle of resource allocation and management.
pub unsafe trait TensorRepr {
    /// Returns the number of dimensions (axes) of the tensor. The same object must return the same number even through any mutable operations (the only exception is `mem::{swap,replace,take}` operations: these ops semantically don't change objects themselves).
    fn dim(&self) -> usize;
}

pub trait AsViewRepr<'a>: TensorRepr {
    type View: TensorRepr;
    fn view(&'a self) -> Self::View;
}

pub trait AsViewMutRepr<'a>: TensorRepr {
    type ViewMut: TensorRepr;
    fn view_mut(&'a mut self) -> Self::ViewMut;
}
