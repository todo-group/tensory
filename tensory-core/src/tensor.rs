//! Layer 2 tensor concept: tensor with axes each indexed with locally unique ID =: legs.

use crate::{
    container::{ContainerImpl, ContainerMapImpl, Raw},
    mapper::{AxisMapper, ReplaceMapper},
    repr::{AsViewMutRepr, AsViewRepr, ReprContext, TensorRepr, TensorTupleRepr},
    task::{Context, IsTask},
};

/// A standard tensor struct.
///
/// In the conceptual model, a tensor is a structured data object with multiple axes, each indexed by a locally unique ID (or "legs").
///
/// In practice, this struct is a wrapper of the compound of a tensor representation and a mapper (`TensorRepr` and `AxisMapper`), ensuring "leg structure". This struct provides safe interfaces to access and modify the underlying representation and mapper. This struct also provides unsafe hatches to access and modify the inners, for implementors of extended functionalitys.
///
/// # Conceptual Model
///
/// According to the conceptual model of this struct, we can rephrase "leg structure" as follows:
///
/// - A tensor has a fixed number of legs, never changed even through mutable operations.
/// - Each leg is indexed by a locally unique ID, never changed even through mutable operations.
/// - The "semantic assignment" to legs is never changed, even through mutable operations.
///
/// ## Semantic Assignment
///
/// !!! DOCUMENT IS WIP !!!
///
/// - "semantic assignment" means the assignment of "semantic meaning" to the legs.
/// - "semantic meaning" means the conceptual meaning of the axis in the context of internal data representation.
/// - for example, let a 5 * 5 * 5 array A_(ijk) has 3 axes, each indexed by ID `X`, `Y`, `Z` in the same order. Here we define the (partially) transposed array B_(ijk) = A_(kji). A_(ijk) and B_(ijk) has same "shape": every corresponding axis pair has same size. therefore the silent replacement of the internal data from A to B is syntactically valid. but semantically the "semantic assignment" is changed from `(X,Y,Z)` to `(Z,Y,X)`. So this operation violates the invariant of "semantic assignment" and is not allowed as a mutable operation of this struct. Instead, this operation can be implemented as a by-value operation returning a new tensor.

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct TensorTuple<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> {
    repr: T,
    mappers: [M; N],
}

impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper> TensorTuple<N, R, M> {
    /// Create a tensor from raw representation and mapper, checking the invariant `repr.naxes() == mapper.naxes()`.
    pub fn from_raw(repr: R, mappers: [M; N]) -> Result<Self, (R, [M; N])> {
        if repr.naxeses() == mappers.each_ref().map(|m| m.naxes()) {
            Ok(unsafe { Self::from_raw_unchecked(repr, mappers) })
        } else {
            Err((repr, mappers))
        }
    }

    /// Create a tensor from raw representation and mapper without checking the invariant `repr.naxes() == mapper.naxes()`.
    ///
    /// # Safety
    ///
    /// caller must ensure the invariant `repr.naxes() == mapper.naxes()`
    pub unsafe fn from_raw_unchecked(repr: R, mappers: [M; N]) -> Self {
        Self { repr, mappers }
    }

    /// Decompose the tensor into raw representation and mapper.
    pub fn into_raw(self) -> (R, [M; N]) {
        (self.repr, self.mappers)
    }

    /// Get the immutable reference to the mapper of the tensor.
    pub fn mappers(&self) -> &[M; N] {
        &self.mappers
    }

    /// Get a mutable reference to the mapper of the tensor.
    ///
    /// # Safety
    ///
    /// The caller MUST NOT swap the object using `std::mem::{swap,replace,take,...}`.
    pub unsafe fn mappers_mut(&mut self) -> &mut [M; N] {
        &mut self.mappers
    }

    /// Get the immutable reference to the representation of the tensor.
    pub fn repr(&self) -> &R {
        &self.repr
    }

    /// Get a mutable reference to the representation of the tensor.
    ///
    /// # Safety
    ///
    /// The caller MUST NOT swap the object using `std::mem::{swap,replace,take,...}`.
    pub unsafe fn repr_mut(&mut self) -> &mut R {
        &mut self.repr
    }

    /// Create a immutable view of the tensor.
    ///
    /// In method chain, you would be able to replace `a.view()` with `(&a)` thanks to boilerplate implementations.
    pub fn view<'a>(&'a self) -> TensorTuple<N, R::View, M>
    where
        M: Clone,
        R: AsViewRepr<'a, N>,
    {
        unsafe { TensorTuple::from_raw_unchecked(self.repr().view(), self.mappers().clone()) }
    }
    /// Create a mutable view of the tensor.
    ///
    /// In method chain, you would be able to replace `a.view_mut()` with `(&mut a)` thanks to boilerplate implementations.
    pub fn view_mut<'a>(&'a mut self) -> TensorTuple<N, R::ViewMut, M>
    where
        M: Clone,
        R: AsViewMutRepr<'a, N>,
    {
        let mapper = self.mappers().clone();
        unsafe { TensorTuple::from_raw_unchecked(self.repr_mut().view_mut(), mapper) }
    }
}

mod private {
    use crate::{mapper::AxisMapper, repr::TensorTupleRepr, tensor::TensorTuple};
    pub trait ToTensorTupleSealed<const N: usize> {}
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> ToTensorTupleSealed<N>
        for TensorTuple<N, T, M>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> ToTensorTupleSealed<N>
        for &TensorTuple<N, T, M>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> ToTensorTupleSealed<N>
        for &mut TensorTuple<N, T, M>
    {
    }
}

/// Conversion trait to Tensor.
///
/// This trait is sealed, and implemented for Tensor, &Tensor, &mut Tensor. &Tensor and &mut Tensor will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for Tensor, &Tensor, &mut Tensor; this is required because `&` operator is not overloadable.
pub trait ToTensorTuple<const N: usize>: private::ToTensorTupleSealed<N> {
    /// The representation type of the resulting tensor.
    type Repr: TensorTupleRepr<N>;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// Converts itself to a tensor.
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper>;
}

impl<T: TensorTupleRepr<N>, M: AxisMapper, const N: usize> ToTensorTuple<N>
    for TensorTuple<N, T, M>
{
    type Repr = T;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self
    }
}
impl<'a, T: AsViewRepr<'a, N>, M: AxisMapper + Clone, const N: usize> ToTensorTuple<N>
    for &'a TensorTuple<N, T, M>
{
    type Repr = T::View;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self.view()
    }
}
impl<'a, T: AsViewMutRepr<'a, N>, M: AxisMapper + Clone, const N: usize> ToTensorTuple<N>
    for &'a mut TensorTuple<N, T, M>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self.view_mut()
    }
}

pub type Tensor<R, M> = TensorTuple<1, R, M>;

impl<R: TensorRepr, M: AxisMapper> Tensor<R, M> {
    /// Get the immutable reference to the mapper of the tensor.
    pub fn mapper(&self) -> &M {
        &self.mappers[0]
    }

    /// Get a mutable reference to the mapper of the tensor.
    ///
    /// # Safety
    ///
    /// The caller MUST NOT swap the object using `std::mem::{swap,replace,take,...}`.
    pub unsafe fn mapper_mut(&mut self) -> &mut M {
        &mut self.mappers[0]
    }
}

/// General utility trait for tensor operations.
pub trait TensorExt: ToTensorTuple<1> {
    /// Replace a ID of a leg of the tensor.
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<TensorTuple<1, Self::Repr, Self::Mapper>, <Self::Mapper as ReplaceMapper<Q>>::Err>
    where
        Self::Mapper: ReplaceMapper<Q>;
}

impl<T: ToTensorTuple<1>> TensorExt for T {
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<TensorTuple<1, Self::Repr, Self::Mapper>, <Self::Mapper as ReplaceMapper<Q>>::Err>
    where
        Self::Mapper: ReplaceMapper<Q>,
    {
        let (repr, [mapper]) = self.to_tensor_tuple().into_raw();
        let mapper = mapper.replace(query)?;
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, [mapper]) })
    }
}

impl<const N: usize, T: TensorTupleRepr<N> + IsTask, M: AxisMapper> IsTask
    for TensorTuple<N, T, M>
{
}

impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, Mk, C: ReprContext<Mk, N, T>>
    Context<Mk, TensorTuple<N, T, M>> for C
where
    C::CType: ContainerMapImpl<C::Repr, TensorTuple<N, C::Repr, M>>,
{
    type Output = <C::CType as ContainerImpl<TensorTuple<N, C::Repr, M>>>::Container;

    fn execute(self, task: TensorTuple<N, T, M>) -> Self::Output {
        let (repr, mappers) = task.into_raw();
        let repr = self.execute(repr);

        C::CType::map(repr, |repr| unsafe {
            TensorTuple::from_raw_unchecked(repr, mappers)
        })
    }
}
pub unsafe trait TensorContext<Mk, const N: usize, T: TensorTupleRepr<N>, M: AxisMapper>:
    Context<
        Mk,
        TensorTuple<N, T, M>,
        Output = <Self::CType as ContainerImpl<TensorTuple<N, Self::Repr, M>>>::Container,
    >
{
    type Repr: TensorTupleRepr<N>;
    type CType: ContainerImpl<TensorTuple<N, Self::Repr, M>>;
}

unsafe impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, Mk, C: ReprContext<Mk, N, T>>
    TensorContext<Mk, N, T, M> for C
where
    C::CType: ContainerMapImpl<
            <Self as ReprContext<Mk, N, T>>::Repr,
            TensorTuple<N, <Self as ReprContext<Mk, N, T>>::Repr, M>,
        >,
{
    type Repr = C::Repr;
    type CType = C::CType;
}

// /// General utility trait for task tensor tuple operations.
// pub trait TaskTensorTupleExt<const N: usize>: ToTensorTuple<N> {
//     fn with<Mk, C: Context<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> TensorTuple<N, C::Res, Self::Mapper>;
//     fn exec<Mk>(self) -> TensorTuple<N, <() as Context<Mk, N, Self::Repr>>::Res, Self::Mapper>
//     where
//         (): Context<Mk, N, Self::Repr>;
// }
// impl<const N: usize, T: ToTensorTuple<N>> TaskTensorTupleExt<N> for T {
//     fn with<Mk, C: Context<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> TensorTuple<N, C::Res, Self::Mapper> {
//         let (repr, mappers) = self.to_tensor_tuple().into_raw();
//         unsafe { TensorTuple::from_raw_unchecked(ctx.execute(repr), mappers) }
//     }
//     fn exec<Mk>(self) -> TensorTuple<N, <() as Context<Mk, N, Self::Repr>>::Res, Self::Mapper>
//     where
//         (): Context<Mk, N, Self::Repr>,
//     {
//         TaskTensorTupleExt::<N>::with(self, ())
//     }
// }

// pub trait TaskTensorTupleOptionExt<const N: usize>: ToTensorTuple<N> {
//     fn with<Mk, C: OptionContext<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> Option<TensorTuple<N, C::Res, Self::Mapper>>;
//     fn exec<Mk>(
//         self,
//     ) -> Option<TensorTuple<N, <() as OptionContext<Mk, N, Self::Repr>>::Res, Self::Mapper>>
//     where
//         (): OptionContext<Mk, N, Self::Repr>;
// }

// impl<const N: usize, T: ToTensorTuple<N>> TaskTensorTupleOptionExt<N> for T {
//     fn with<Mk, C: OptionContext<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> Option<TensorTuple<N, C::Res, Self::Mapper>> {
//         let (repr, mappers) = self.to_tensor_tuple().into_raw();
//         Some(unsafe { TensorTuple::from_raw_unchecked(ctx.execute(repr)?, mappers) })
//     }
//     fn exec<Mk>(
//         self,
//     ) -> Option<TensorTuple<N, <() as OptionContext<Mk, N, Self::Repr>>::Res, Self::Mapper>>
//     where
//         (): OptionContext<Mk, N, Self::Repr>,
//     {
//         TaskTensorTupleOptionExt::<N>::with(self, ())
//     }
// }

// pub trait TaskTensorTupleResultExt<const N: usize>: ToTensorTuple<N> {
//     fn with<Mk, C: ResultContext<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> Result<TensorTuple<N, C::Res, Self::Mapper>, C::Err>;
//     fn exec<Mk>(
//         self,
//     ) -> Result<
//         TensorTuple<N, <() as ResultContext<Mk, N, Self::Repr>>::Res, Self::Mapper>,
//         <() as ResultContext<Mk, N, Self::Repr>>::Err,
//     >
//     where
//         (): ResultContext<Mk, N, Self::Repr>;
// }

// impl<const N: usize, T: ToTensorTuple<N>> TaskTensorTupleResultExt<N> for T {
//     fn with<Mk, C: ResultContext<Mk, N, Self::Repr>>(
//         self,
//         ctx: C,
//     ) -> Result<TensorTuple<N, C::Res, Self::Mapper>, C::Err> {
//         let (repr, mappers) = self.to_tensor_tuple().into_raw();
//         Ok(unsafe { TensorTuple::from_raw_unchecked(ctx.execute(repr)?, mappers) })
//     }
//     fn exec<Mk>(
//         self,
//     ) -> Result<
//         TensorTuple<N, <() as ResultContext<Mk, N, Self::Repr>>::Res, Self::Mapper>,
//         <() as ResultContext<Mk, N, Self::Repr>>::Err,
//     >
//     where
//         (): ResultContext<Mk, N, Self::Repr>,
//     {
//         TaskTensorTupleResultExt::<N>::with(self, ())
//     }
// }

// /// Trait expressing a tensor operation "task" that can be executed with a context.
// ///
// /// Tensory uses a "task" and "context" model for expressing tensor operations. A "task" represents an abstract operation defined in axis/leg level description. The implementation of the "task" is implemented on "context". Here, you can consume the tensor and "context" for the operation. Therefore "context" can be considered as a generalized "Strategy" design pattern, which are allowed to hold own resources.
// ///
// /// In practice, it is RECOMMENDED to execute mappers decomposition/reconstruction in the construction of the "task", and to delegate the operation on representation to "context".
// pub trait TensorTask<C> {
//     /// The output type of the task.
//     type Output;
//     /// Executes the task, delegating the implementation to `ctx`.
//     fn with(self, ctx: C) -> Self::Output;
// }

// /// Utility trait for executing a tensor task with the default context `()`.
// pub trait TensorDefaultTask: TensorTask<()> {
//     /// Executes the task with the default context `()`.
//     fn exec(self) -> Self::Output;
// }
// impl<T: TensorTask<()>> TensorDefaultTask for T {
//     fn exec(self) -> Self::Output {
//         self.with(())
//     }
// }

// struct TensorMutRefGuard<'a, M:AxisMgr, T:TensorRepr> {
//     raw: &'a mut T,
//     mgr: &'a mut M,
// }

// impl<'a, M: AxisMgr, T: TensorRepr> Drop for TensorMutRefGuard<'a, M, T> {
//     fn drop(&mut self) {
//     self.raw
//         // self.mgr.borrow_mut().use_mut();
//         // self.raw.use_mut();
//         // self.mgr.use_mut();
//     }
// }

// impl<'a, M, T> TensorMutRefGuard<'a, M, T> {
//     f
// }

// #[cfg(test)]
// mod tests {

//     use std::println;

//     use crate::leg;

//     use super::*;

//     #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
//     struct DummyTensor(usize);
//     unsafe impl TensorRepr for DummyTensor {
//         fn dim(&self) -> usize {
//             self.0
//         }
//     }

//     #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
//     struct DummyLegId;

//     #[test]
//     fn it_works() {
//         let raw_tensor = DummyTensor(1);

//         let ts = Tensor::from_raw(raw_tensor).unwrap();

//         println!("{:?}", ts.mapper());
//     }
// }

// pub enum MonadSignature {
//     None,
//     Option,
//     Result,
// }

// pub trait MonadMarker<const SIG: MonadSignature> {}

// pub unsafe trait TensorContext<M, T: TensorRepr>:
//     Context<M, T, Res = <Self as TensorContext<M, T>>::Res> + MonadMarker<const SIG: MonadSignature = MonadSignature::None>
// {
//     type Res: TensorRepr;
// }
// pub unsafe trait TensorOptionContext<M, T: TensorRepr>:
//     Context<M, T, Res = Option<<Self as TensorOptionContext<M, T>>::Res>>
//     + MonadMarker<SIG = MonadSignature::Option>
// {
//     type Res: TensorRepr;
// }
// pub unsafe trait TensorResultContext<M, T: TensorRepr>:
//     Context<
//         M,
//         T,
//         Res = Result<
//             <Self as TensorResultContext<M, T>>::Res,
//             <Self as TensorResultContext<M, T>>::Err,
//         >,
//     >
// {
//     type Res: TensorRepr;
//     type Err;
// }

// pub unsafe trait TensorContext<Mk, const N: usize, T: TensorTupleRepr<N>>:
//     Context<Mk, T, Res: ReprContainer<N>>
// {
// }

// impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, Mk, C: TensorContext<Mk, N, T>>
//     TaskDelegate<Mk, T, C> for TensorTuple<N, T, M>
// {
//     type Output ;

//     fn with(self, ctx: C) -> Self::Output
//     where
//         Self: Sized,
//     {
//         todo!()
//     }
// }
