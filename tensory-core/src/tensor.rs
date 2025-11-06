//! Layer 2 tensor concept: tensor with axes each indexed with locally unique ID =: legs.

use crate::{
    mapper::{AxisMapper, ReplaceMapper},
    repr::{AsViewMutRepr, AsViewRepr, TensorRepr},
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
pub struct Tensor<T: TensorRepr, M: AxisMapper> {
    repr: T,
    mapper: M,
}

impl<R: TensorRepr, M: AxisMapper> Tensor<R, M> {
    /// Create a tensor from raw representation and mapper, checking the invariant `repr.naxes() == mapper.naxes()`.
    pub fn from_raw(repr: R, mapper: M) -> Result<Self, (R, M)> {
        if repr.naxes() == mapper.naxes() {
            Ok(unsafe { Self::from_raw_unchecked(repr, mapper) })
        } else {
            Err((repr, mapper))
        }
    }

    /// Create a tensor from raw representation and mapper without checking the invariant `repr.naxes() == mapper.naxes()`.
    ///
    /// # Safety
    ///
    /// caller must ensure the invariant `repr.naxes() == mapper.naxes()`
    pub unsafe fn from_raw_unchecked(repr: R, mapper: M) -> Self {
        Self { repr, mapper }
    }

    /// Decompose the tensor into raw representation and mapper.
    pub fn into_raw(self) -> (R, M) {
        (self.repr, self.mapper)
    }

    /// Get the immutable reference to the mapper of the tensor.
    pub fn mapper(&self) -> &M {
        &self.mapper
    }

    /// Get a mutable reference to the mapper of the tensor.
    ///
    /// # Safety
    ///
    /// The caller MUST NOT swap the object using `std::mem::{swap,replace,take,...}`.
    pub unsafe fn mapper_mut(&mut self) -> &mut M {
        &mut self.mapper
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
    pub fn view<'a>(&'a self) -> Tensor<R::View, M>
    where
        M: Clone,
        R: AsViewRepr<'a>,
    {
        unsafe { Tensor::from_raw_unchecked(self.repr().view(), self.mapper().clone()) }
    }
    /// Create a mutable view of the tensor.
    ///
    /// In method chain, you would be able to replace `a.view_mut()` with `(&mut a)` thanks to boilerplate implementations.
    pub fn view_mut<'a>(&'a mut self) -> Tensor<R::ViewMut, M>
    where
        M: Clone,
        R: AsViewMutRepr<'a>,
    {
        let mapper = self.mapper().clone();
        unsafe { Tensor::from_raw_unchecked(self.repr_mut().view_mut(), mapper) }
    }
}

/// General utility trait for tensor operations.
pub trait TensorExt: ToTensor {
    /// Replace a ID of a leg of the tensor.
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<Tensor<Self::Repr, Self::Mapper>, <Self::Mapper as ReplaceMapper<Q>>::Err>
    where
        Self::Mapper: ReplaceMapper<Q>;
}

impl<T: ToTensor> TensorExt for T {
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<Tensor<Self::Repr, Self::Mapper>, <Self::Mapper as ReplaceMapper<Q>>::Err>
    where
        Self::Mapper: ReplaceMapper<Q>,
    {
        let (repr, mapper) = self.to_tensor().into_raw();
        let mapper = mapper.replace(query)?;
        Ok(unsafe { Tensor::from_raw_unchecked(repr, mapper) })
    }
}

mod private {
    use crate::{mapper::AxisMapper, repr::TensorRepr, tensor::Tensor};
    pub trait ToTensorSealed {}
    impl<T: TensorRepr, M: AxisMapper> ToTensorSealed for Tensor<T, M> {}
    impl<T: TensorRepr, M: AxisMapper> ToTensorSealed for &Tensor<T, M> {}
    impl<T: TensorRepr, M: AxisMapper> ToTensorSealed for &mut Tensor<T, M> {}
}

/// Conversion trait to Tensor.
///
/// This trait is sealed, and implemented for Tensor, &Tensor, &mut Tensor. &Tensor and &mut Tensor will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for Tensor, &Tensor, &mut Tensor; this is required because `&` operator is not overloadable.
pub trait ToTensor: private::ToTensorSealed {
    /// The representation type of the resulting tensor.
    type Repr: TensorRepr;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// Converts itself to a tensor.
    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper>;
}

impl<T: TensorRepr, M: AxisMapper> ToTensor for Tensor<T, M> {
    type Repr = T;
    type Mapper = M;
    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper> {
        self
    }
}
impl<'a, T: AsViewRepr<'a>, M: AxisMapper + Clone> ToTensor for &'a Tensor<T, M> {
    type Repr = T::View;
    type Mapper = M;
    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper> {
        self.view()
    }
}
impl<'a, T: AsViewMutRepr<'a>, M: AxisMapper + Clone> ToTensor for &'a mut Tensor<T, M> {
    type Repr = T::ViewMut;
    type Mapper = M;
    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper> {
        self.view_mut()
    }
}

/// Trait expressing a tensor operation "task" that can be executed with a context.
///
/// Tensory uses a "task" and "context" model for expressing tensor operations. A "task" represents an abstract operation defined in axis/leg level description. The implementation of the "task" is implemented on "context". Here, you can consume the tensor and "context" for the operation. Therefore "context" can be considered as a generalized "Strategy" design pattern, which are allowed to hold own resources.
///
/// In practice, it is RECOMMENDED to execute mappers decomposition/reconstruction in the construction of the "task", and to delegate the operation on representation to "context".
pub trait TensorTask<C> {
    /// The output type of the task.
    type Output;
    /// Executes the task, delegating the implementation to `ctx`.
    fn with(self, ctx: C) -> Self::Output;
}

/// Utility trait for executing a tensor task with the default context `()`.
pub trait TensorDefaultTask: TensorTask<()> {
    /// Executes the task with the default context `()`.
    fn exec(self) -> Self::Output;
}
impl<T: TensorTask<()>> TensorDefaultTask for T {
    fn exec(self) -> Self::Output {
        self.with(())
    }
}

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
