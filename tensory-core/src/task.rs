//! The "task" and "context" model for syntax suger of tensor operations. A "task" is aimed to represent an abstract operation defined in axis/leg level description. The implementation of the "task" is implemented on "context". Here, you can consume the tensor and "context" for the operation. Therefore "context" can be considered as a generalized "Strategy" design pattern, which are allowed to hold own resources.

/// Context object. execute the task in its own way, and return the result. The context can hold its own resources, and can be considered as a generalized "Strategy" design pattern.
/// The parameter Mk is a dummy marker parameter to enable the implemnetaion by the third party.
pub trait Context<Mk, T> {
    /// The type of the result returned by the context.
    type Output;
    /// Executes the task, and return the result.
    fn execute(self, task: T) -> Self::Output;
}

/// enable task oriented syntax suger implemented in `TaskExt`.
pub trait IsTask {}

/// The implementation of the task oriented syntax suger.
pub trait TaskExt: IsTask {
    /// Executes the task with the context `ctx`.
    fn with<Mk, C: Context<Mk, Self>>(self, ctx: C) -> C::Output
    where
        Self: Sized;
    /// Executes the task with the default context `()`.
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

// // T means the core Task passed to context
// pub trait TaskHolder<T = Self> {}

// /// Trait expressing a tensor operation "task" that can be executed with a context.
// ///
// /// Tensory uses a "task" and "context" model for expressing tensor operations. A "task" represents an abstract operation defined in axis/leg level description. The implementation of the "task" is implemented on "context". Here, you can consume the tensor and "context" for the operation. Therefore "context" can be considered as a generalized "Strategy" design pattern, which are allowed to hold own resources.
// ///
// /// In practice, it is RECOMMENDED to execute mappers decomposition/reconstruction in the construction of the "task", and to delegate the operation on representation to "context".

// // T means the core taask passed to the context C
// pub trait TaskDelegate<Mk, T, C: Context<Mk, T>>: TaskHolder<T> {
//     type Output;
//     /// Executes the task, delegating the implementation to `ctx`.
//     fn with(self, ctx: C) -> Self::Output
//     where
//         Self: Sized;
//     // /// Executes the task with the default context `()`.
//     // fn exec<M>(self) -> <() as Context<M, Self>>::Output
//     // where
//     //     (): Context<M, Self>,
//     //     Self: Sized;
// }
// impl<Mk, T: TaskHolder<T>, C: Context<Mk, T>> TaskDelegate<Mk, T, C> for T {
//     type Output = C::Res;
//     fn with(self, ctx: C) -> Self::Output {
//         ctx.execute(self)
//     }
//     // fn exec<M>(self) -> <() as Context<M, Self>>::Output
//     // where
//     //     (): Context<M, Self>,
//     // {
//     //     self.with(())
//     // }
// }
