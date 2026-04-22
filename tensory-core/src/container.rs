use core::marker::PhantomData;

pub trait ContainerType {}

pub struct Raw;
impl ContainerType for Raw {}

pub struct Optioning<C>(PhantomData<C>);
impl<C: ContainerType> ContainerType for Optioning<C> {}

pub struct Resulting<C, E>(PhantomData<C>, PhantomData<E>);
impl<C: ContainerType, E> ContainerType for Resulting<C, E> {}

pub trait ContainerImpl<X>: ContainerType {
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

pub trait ContainerMapImpl<X, Y>: ContainerImpl<X> + ContainerImpl<Y> {
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

pub trait ContainerChainImpl<C: ContainerType>: ContainerType {
    type ContainerType;
}
