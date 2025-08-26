/// Minimal requirement to be a tensor representation.
///
/// This trait requires
/// In mental model, tensor representation is a structured data with multiple dimensions (axes).
///
/// In practice, a tensor representation is treated as a handle of resource allocation and management.
pub unsafe trait TensorRepr {
    /// Returns the number of dimensions (axes) of the tensor. The same object must return the same number even through any mutable operations (the only exception is `mem::{swap,replace,take}` operations: these ops semantically don't change objects themselves).
    fn dim(&self) -> usize;
}
