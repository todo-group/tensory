//! Task execution and runtime-binding abstractions.
//!
//! Tasks are executed through contexts, and bindable values can be associated
//! with runtimes.

use thiserror::Error;

/// Marks a type as a task.
///
/// [`TaskExt`] provides syntax sugar for marked types.
pub trait IsTask {}

/// A strategy for executing a task that may own additional resources.
///
/// The context determines the result type. `Mk` is a local marker used by the
/// Stepparent Pattern to permit implementations in third-party crates.
///
/// # Type Parameters
///
/// * `Mk` identifies the implementation.
/// * `T` is the task.
pub trait Context<Mk, T: IsTask> {
    /// The type of the result returned by the context.
    type Output;
    /// Executes `task` using this context.
    fn execute(self, task: T) -> Self::Output;
}

/// Provides syntax sugar for executing [`IsTask`] types.
pub trait TaskExt: IsTask {
    /// Executes the task with an explicitly supplied context.
    ///
    /// `Mk` identifies the [`Context`] implementation. Specify it as
    /// `with::<Mk, _>(ctx)` when it cannot be inferred.
    fn with<Mk, C: Context<Mk, Self>>(self, ctx: C) -> C::Output
    where
        Self: Sized;
    /// Executes the task with the unit context `()`.
    ///
    /// Uses `()` as the context. `Mk` has the same role as in [`TaskExt::with`];
    /// specify it as `exec::<Mk>()` when it cannot be inferred.
    fn exec<Mk>(self) -> <() as Context<Mk, Self>>::Output
    where
        (): Context<Mk, Self>,
        Self: Sized;
}
impl<T: IsTask> TaskExt for T {
    fn with<Mk, C: Context<Mk, Self>>(self, ctx: C) -> C::Output
    where
        Self: Sized,
    {
        ctx.execute(self)
    }
    fn exec<Mk>(self) -> <() as Context<Mk, Self>>::Output
    where
        (): Context<Mk, Self>,
    {
        self.with(())
    }
}

/// Marks a runtime.
///
/// `Clone` and `Eq` must preserve runtime identity. A cheap `Clone`
/// implementation is recommended.
pub unsafe trait IsRuntime: Clone + Eq {}

/// Provides a context for a task.
///
/// An implementation associates a marker and a context type with `T`.
/// [`RuntimeFor::ctx`] provides that context to users of the runtime.
pub trait RuntimeFor<T: IsTask>: IsRuntime {
    /// Marker for the context provided for `T`.
    type Mk;
    /// Context for processing `T`.
    type Ctx: Context<Self::Mk, T>;
    /// Provides the context for `T`.
    fn ctx(&self) -> Self::Ctx;
}

/// Errors from runtime-bound task operations.
///
/// An operation that obtains a context from a runtime and executes a task in one
/// call may fail due to a runtime mismatch, while deferring the operation, or
/// while executing the task. These cases are represented by [`Self::Runtime`],
/// [`Self::Defer`], and [`Self::Execute`], respectively.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Error)]
pub enum RuntimeErr<DE, EE> {
    /// Runtime mismatch.
    #[error("Runtime mismatch")]
    Runtime,
    /// Error raised while deferring an operation to the runtime.
    #[error("Defer error: {0}")]
    Defer(DE),
    /// Error raised while executing an operation on the runtime.
    #[error("Execute error: {0}")]
    Execute(EE),
}

/// Marks a type that may be stored in [`BoundObject`].
pub trait IsBindable {}

/// Pairs a bindable value with a runtime.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct BoundObject<T: IsBindable, RT: IsRuntime> {
    inner: T,
    runtime: RT,
}

impl<T: IsBindable, RT: IsRuntime> BoundObject<T, RT> {
    /// Creates a runtime-bound object from its object and runtime parts.
    pub fn from_raw(inner: T, runtime: RT) -> Self {
        Self { inner, runtime }
    }
    /// Decomposes the bound object into its object and runtime parts.
    pub fn into_raw(self) -> (T, RT) {
        (self.inner, self.runtime)
    }

    /// Returns an immutable reference to the bound object.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Returns a mutable reference to the bound object.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Returns an immutable reference to the associated runtime.
    pub fn runtime(&self) -> &RT {
        &self.runtime
    }
    // /// Returns a mutable reference to the associated runtime.
    // pub fn runtime_mut(&mut self) -> &mut RT {
    //     &mut self.runtime
    // }
}

/// Provides syntax sugar for binding [`IsBindable`] types to a runtime.
pub trait BoundableExt: IsBindable {
    /// Binds the object with a runtime, producing a runtime-bound object.
    fn bind<RT: IsRuntime>(self, runtime: RT) -> BoundObject<Self, RT>
    where
        Self: Sized;
}
impl<T: IsBindable> BoundableExt for T {
    fn bind<RT: IsRuntime>(self, runtime: RT) -> BoundObject<Self, RT> {
        BoundObject::from_raw(self, runtime)
    }
}
impl<T: IsBindable, RT: IsRuntime> BoundObject<T, RT> {
    /// Removes the runtime and returns the underlying object.
    pub fn unbind(self) -> T {
        self.into_raw().0
    }
}
