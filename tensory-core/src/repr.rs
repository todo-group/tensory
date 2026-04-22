//! Layer 1 tensor concept: tensor with axes 0-indexed with usize, only valid in the tensor.

use crate::{container::ContainerImpl, task::Context};

/// Minimal interface for a tuple of tensor representations.
///
/// In the conceptual model, a tensor representation is a structured data object with multiple axes, each indexed from `0` up to `naxes - 1`.
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
pub unsafe trait TensorTupleRepr<const N: usize>: Sized {
    /// Returns the number of axes of the tensor. this number is fixed for the same object even through mutable operations.
    ///
    /// this function serves as a dynamic version of `const N:usize`.
    fn naxeses(&self) -> [usize; N];
}

/// Backward compatibility: a tensor representation is now treated as a tuple of tensor representations with one element.
pub unsafe trait TensorRepr: TensorTupleRepr<1> {
    /// Returns the number of axes of the tensor. this number is fixed for the same object even through mutable operations.
    ///
    /// this function serves as a dynamic version of `const N: usize`.
    fn naxes(&self) -> usize;
}
unsafe impl<T: TensorTupleRepr<1>> TensorRepr for T {
    fn naxes(&self) -> usize {
        let [n] = self.naxeses();
        n
    }
}

/// Interface to generate a immutable view representation of itself.
///
/// # Safety
///
/// The implementor MUST ensure that the view representations have the same semantic structure of axes as the original representation.
pub unsafe trait AsViewRepr<'a, const N: usize>: TensorTupleRepr<N> {
    /// Immutable view representation type.
    type View: TensorTupleRepr<N>;
    /// Returns an immutable view representation of itself. The view has the same semantic structure of axes as the original representation.
    fn view(&'a self) -> Self::View;
}

/// Interface to generate a mutable view representation of itself.
///
/// # Safety
///
/// The implementor MUST ensure that the view representation has the same semantic structure of axes as the original representation.
pub unsafe trait AsViewMutRepr<'a, const N: usize>: TensorTupleRepr<N> {
    /// Mutable view representation type.
    type ViewMut: TensorTupleRepr<N>;
    /// Returns a mutable view representation of itself. The view has the same semantic structure of axes as the original representation.
    fn view_mut(&'a mut self) -> Self::ViewMut;
}

unsafe impl<A: TensorTupleRepr<1>, B: TensorTupleRepr<1>> TensorTupleRepr<2> for (A, B) {
    fn naxeses(&self) -> [usize; 2] {
        [self.0.naxes(), self.1.naxes()]
    }
}
unsafe impl<A: TensorTupleRepr<1>, B: TensorTupleRepr<1>, C: TensorTupleRepr<1>> TensorTupleRepr<3>
    for (A, B, C)
{
    fn naxeses(&self) -> [usize; 3] {
        [self.0.naxes(), self.1.naxes(), self.2.naxes()]
    }
}

/// Marker trait that indicates the context keeps the semantic structure of the tensor.
///
/// # Safety
///
/// The implementor must ensure that the context keeps the semantic structure of the tensor, which means the result tensor must have the same axis structure as the input tensor, in the implementation of Context<Mk, T>.
pub unsafe trait ReprContext<Mk, const N: usize, T: TensorTupleRepr<N>>:
    Context<Mk, T, Output = <Self::CType as ContainerImpl<Self::Repr>>::Container>
{
    type Repr: TensorTupleRepr<N>;
    type CType: ContainerImpl<Self::Repr>;
}
