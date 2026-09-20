//! Concepts for tensor representations and the forms that support them.
//!
//! This module defines representations of one or more values, together with
//! their immutable, mutable, and owned forms and the context contract for
//! operations that preserve them. A representation's semantic axis order is
//! the correspondence between an axis's position and its conceptual identity;
//! preserving it prevents an operation from silently reinterpreting one axis
//! as another.

use crate::concept::{
    container::ContainerImpl,
    task::{Context, IsTask},
};

/// Describes the representation of a fixed-size tuple of values.
///
/// `N` determines the number of represented values. The values may have
/// different concrete representation types.
///
/// # Safety
///
/// Implementations must preserve the number and semantic axis order of every
/// represented value through mutable operations.
pub unsafe trait TensorTupleRepr<const N: usize>: Sized {
    /// Returns the number of axes for each represented value.
    fn naxes_array(&self) -> [usize; N];
}

/// Syntax sugar for [`TensorTupleRepr<1>`].
///
/// It provides the single-representation form and its dynamic axis count.
pub trait TensorRepr: TensorTupleRepr<1> {
    /// Returns the number of axes.
    ///
    /// This is the runtime counterpart of the tuple's const `N` parameter.
    fn naxes(&self) -> usize;
}
impl<R: TensorTupleRepr<1>> TensorRepr for R {
    fn naxes(&self) -> usize {
        let [naxes] = self.naxes_array();
        naxes
    }
}

unsafe impl<A: TensorRepr, B: TensorRepr> TensorTupleRepr<2> for (A, B) {
    fn naxes_array(&self) -> [usize; 2] {
        [self.0.naxes(), self.1.naxes()]
    }
}

unsafe impl<A: TensorRepr, B: TensorRepr, C: TensorRepr> TensorTupleRepr<3> for (A, B, C) {
    fn naxes_array(&self) -> [usize; 3] {
        [self.0.naxes(), self.1.naxes(), self.2.naxes()]
    }
}

/// Provides an immutable view of a representation.
///
/// # Safety
///
/// The view must preserve the number and semantic axis order of every
/// represented value.
pub unsafe trait AsViewRepr<'a, const N: usize>: TensorTupleRepr<N> {
    /// Immutable view type.
    type View: TensorTupleRepr<N>;
    /// Returns an immutable view of every represented value.
    fn view(&'a self) -> Self::View;
}

/// Provides a mutable view of a representation.
///
/// # Safety
///
/// The mutable view must preserve the number and semantic axis order of every
/// represented value.
pub unsafe trait AsViewMutRepr<'a, const N: usize>: TensorTupleRepr<N> {
    /// Mutable view type.
    type ViewMut: TensorTupleRepr<N>;
    /// Returns a mutable view of every represented value.
    fn view_mut(&'a mut self) -> Self::ViewMut;
}

/// Provides an owned form of a representation.
///
/// # Safety
///
/// The owned representation must preserve the number and semantic axis order
/// of every represented value.
pub unsafe trait IntoOwnedRepr<const N: usize>: TensorTupleRepr<N> {
    /// Owned representation type.
    type Owned: TensorTupleRepr<N>;
    /// Converts the representation into its owned form.
    fn into_owned(self) -> Self::Owned;
}

/// Marks a context whose output preserves the representation structure of its
/// task.
///
/// `Self::Repr` is the representation produced by the context, and `CType`
/// describes how it is wrapped in the context's output.
///
/// # Type Parameters
///
/// * `Mk` identifies the underlying [`Context`] implementation.
/// * `N` is the number of represented values.
/// * `T` is the task whose representation is preserved.
///
/// # Safety
///
/// The [`Context::execute`] implementation must preserve the number and
/// semantic axis order of `T` in every representation it produces.
pub unsafe trait ReprContext<Mk, const N: usize, T: TensorTupleRepr<N> + IsTask>:
    Context<Mk, T, Output = <Self::CType as ContainerImpl<Self::Repr>>::Container>
{
    /// Representation produced by the context.
    type Repr: TensorTupleRepr<N>;
    /// Container description for the context output.
    type CType: ContainerImpl<Self::Repr>;
}
