//! Type-level descriptions of result containers.
//!
//! A container description separates a payload type from its wrapper structure.
//! The structure can be resolved to a concrete type or preserved while mapping
//! the payload.

use core::marker::PhantomData;

/// Marks a type as a container description.
pub trait ContainerType {}

/// Describes an unwrapped payload.
pub struct Raw;
impl ContainerType for Raw {}

/// Describes an [`Option`] containing another container description.
pub struct Optioning<C>(PhantomData<C>);
impl<C: ContainerType> ContainerType for Optioning<C> {}

/// Describes a [`Result`] containing another container description.
pub struct Resulting<C, E>(PhantomData<C>, PhantomData<E>);
impl<C: ContainerType, E> ContainerType for Resulting<C, E> {}

/// Resolves a container description for a payload type.
///
/// `Raw` yields `X`, `Optioning<C>` yields
/// `Option<<C as ContainerImpl<X>>::Container>`, and `Resulting<C, E>` yields
/// `Result<<C as ContainerImpl<X>>::Container, E>`.
pub trait ContainerImpl<X>: ContainerType {
    /// The concrete container type for `X`.
    type Container;
}
impl<X> ContainerImpl<X> for Raw {
    type Container = X;
}
impl<X, C: ContainerImpl<X>> ContainerImpl<X> for Optioning<C> {
    type Container = Option<C::Container>;
}
impl<X, C: ContainerImpl<X>, E> ContainerImpl<X> for Resulting<C, E> {
    type Container = Result<C::Container, E>;
}

/// Maps a payload without changing the container shape.
///
/// `None` and `Err` are preserved without calling `f`.
pub trait ContainerMapImpl<X, Y>: ContainerImpl<X> + ContainerImpl<Y> {
    /// Applies `f` to the payload in `from`.
    fn map<F: FnOnce(X) -> Y>(
        from: <Self as ContainerImpl<X>>::Container,
        f: F,
    ) -> <Self as ContainerImpl<Y>>::Container;
}
impl<X, Y> ContainerMapImpl<X, Y> for Raw {
    fn map<F: FnOnce(X) -> Y>(from: X, f: F) -> Y {
        f(from)
    }
}
impl<X, Y, C: ContainerMapImpl<X, Y>> ContainerMapImpl<X, Y> for Optioning<C> {
    fn map<F: FnOnce(X) -> Y>(
        from: Option<<C as ContainerImpl<X>>::Container>,
        f: F,
    ) -> Option<<C as ContainerImpl<Y>>::Container> {
        from.map(|c| C::map(c, f))
    }
}
impl<X, Y, C: ContainerMapImpl<X, Y>, E> ContainerMapImpl<X, Y> for Resulting<C, E> {
    fn map<F: FnOnce(X) -> Y>(
        from: Result<<C as ContainerImpl<X>>::Container, E>,
        f: F,
    ) -> Result<<C as ContainerImpl<Y>>::Container, E> {
        from.map(|c| C::map(c, f))
    }
}

// pub trait ContainerChainImpl<C: ContainerType>: ContainerType {
//     type ContainerType;
// }
