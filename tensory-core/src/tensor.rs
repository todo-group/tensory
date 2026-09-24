//! Layer 2 tensor concept: tensor with axes each indexed with locally unique ID =: legs.

use crate::{
    concept::{
        container::{ContainerImpl, ContainerMapImpl},
        task::{BoundObject, Context, IsBindable, IsRuntime, IsTask},
    },
    mapper::{AxisMapper, ReplaceMapper},
    repr::{AsViewMutRepr, AsViewRepr, IntoOwnedRepr, ReprContext, TensorRepr, TensorTupleRepr},
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
pub struct TensorTuple<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper> {
    repr: R,
    mappers: [M; N],
}

/// A single tensor represented as a one-element tensor tuple.
pub type Tensor<R, M> = TensorTuple<1, R, M>;

impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper> TensorTuple<N, R, M> {
    /// Creates a tensor tuple after checking every representation/mapper pair.
    pub fn from_raw(repr: R, mappers: [M; N]) -> Result<Self, (R, [M; N])> {
        if repr
            .naxes_array()
            .into_iter()
            .zip(mappers.iter())
            .all(|(naxes, mapper)| naxes == mapper.naxes())
        {
            Ok(unsafe { Self::from_raw_unchecked(repr, mappers) })
        } else {
            Err((repr, mappers))
        }
    }

    /// Creates a tensor tuple without checking representation/mapper pairs.
    ///
    /// # Safety
    ///
    /// For every index `i`, `repr.naxes_array()[i]` must equal
    /// `mappers[i].naxes()`.
    pub unsafe fn from_raw_unchecked(repr: R, mappers: [M; N]) -> Self {
        Self { repr, mappers }
    }

    /// Decomposes the tensor tuple into its representation and mappers.
    pub fn into_raw(self) -> (R, [M; N]) {
        (self.repr, self.mappers)
    }

    /// Returns the mappers for all represented tensors.
    pub fn mapper_array(&self) -> &[M; N] {
        &self.mappers
    }

    pub unsafe fn mapper_array_mut(&mut self) -> &mut [M; N] {
        &mut self.mappers
    }

    /// Returns a shared reference to the representation bundle.
    pub fn repr(&self) -> &R {
        &self.repr
    }

    /// Returns a mutable reference to the representation bundle.
    ///
    /// # Safety
    ///
    /// The caller must preserve the number and semantic axis structure of all
    /// represented tensors.
    pub unsafe fn repr_mut(&mut self) -> &mut R {
        &mut self.repr
    }
}

impl<R: TensorTupleRepr<1>, M: AxisMapper> TensorTuple<1, R, M> {
    /// Get the immutable reference to the mapper of the tensor.
    pub fn mapper(&self) -> &M {
        let [mapper] = &self.mappers;
        mapper
    }

    /// Get a mutable reference to the mapper of the tensor.
    ///
    /// # Safety
    ///
    /// The caller MUST NOT swap the object using `std::mem::{swap,replace,take,...}`.
    pub unsafe fn mapper_mut(&mut self) -> &mut M {
        let [mapper] = &mut self.mappers;
        mapper
    }
}

impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper> TensorTuple<N, R, M> {
    /// Creates an immutable view of every represented tensor.
    pub fn view<'a>(&'a self) -> TensorTuple<N, R::View, M>
    where
        M: Clone,
        R: AsViewRepr<'a, N>,
    {
        unsafe { TensorTuple::from_raw_unchecked(self.repr.view(), self.mappers.clone()) }
    }

    /// Creates a mutable view of every represented tensor.
    pub fn view_mut<'a>(&'a mut self) -> TensorTuple<N, R::ViewMut, M>
    where
        M: Clone,
        R: AsViewMutRepr<'a, N>,
    {
        let mappers = self.mappers.clone();
        unsafe { TensorTuple::from_raw_unchecked(self.repr.view_mut(), mappers) }
    }

    /// Creates an owned version of every represented tensor.
    pub fn into_owned(self) -> TensorTuple<N, R::Owned, M>
    where
        R: IntoOwnedRepr<N>,
    {
        let (repr, mappers) = self.into_raw();
        unsafe { TensorTuple::from_raw_unchecked(repr.into_owned(), mappers) }
    }
}

impl<A: TensorTupleRepr<1>, B: TensorTupleRepr<1>, M: AxisMapper> TensorTuple<2, (A, B), M> {
    /// Splits a two-output tensor tuple into ordinary tensors.
    pub fn unpack(self) -> (Tensor<A, M>, Tensor<B, M>) {
        let ((a, b), [a_mapper, b_mapper]) = self.into_raw();
        unsafe {
            (
                Tensor::from_raw_unchecked(a, [a_mapper]),
                Tensor::from_raw_unchecked(b, [b_mapper]),
            )
        }
    }

    /// Splits the tuple into two ordinary tensors.
    pub fn pack((a, b): (Tensor<A, M>, Tensor<B, M>)) -> Self {
        let (a_repr, [a_mapper]) = a.into_raw();
        let (b_repr, [b_mapper]) = b.into_raw();
        unsafe { TensorTuple::from_raw_unchecked((a_repr, b_repr), [a_mapper, b_mapper]) }
    }
}

impl<A: TensorTupleRepr<1>, B: TensorTupleRepr<1>, C: TensorTupleRepr<1>, M: AxisMapper>
    TensorTuple<3, (A, B, C), M>
{
    /// Splits a three-output tensor tuple into ordinary tensors.
    pub fn unpack(self) -> (Tensor<A, M>, Tensor<B, M>, Tensor<C, M>) {
        let ((a, b, c), [a_mapper, b_mapper, c_mapper]) = self.into_raw();
        unsafe {
            (
                Tensor::from_raw_unchecked(a, [a_mapper]),
                Tensor::from_raw_unchecked(b, [b_mapper]),
                Tensor::from_raw_unchecked(c, [c_mapper]),
            )
        }
    }

    /// Splits the tuple into three ordinary tensors.
    pub fn pack((a, b, c): (Tensor<A, M>, Tensor<B, M>, Tensor<C, M>)) -> Self {
        let (a_repr, [a_mapper]) = a.into_raw();
        let (b_repr, [b_mapper]) = b.into_raw();
        let (c_repr, [c_mapper]) = c.into_raw();
        unsafe {
            TensorTuple::from_raw_unchecked(
                (a_repr, b_repr, c_repr),
                [a_mapper, b_mapper, c_mapper],
            )
        }
    }
}

pub trait PackExt<const N: usize> {
    type Repr: TensorTupleRepr<N>;
    type Mapper: AxisMapper;
    fn pack(self) -> TensorTuple<N, Self::Repr, Self::Mapper>;
}
impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> PackExt<N> for TensorTuple<N, T, M> {
    type Repr = T;
    type Mapper = M;
    fn pack(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self
    }
}
impl<T1: TensorRepr, T2: TensorRepr, M: AxisMapper> PackExt<2> for (Tensor<T1, M>, Tensor<T2, M>) {
    type Repr = (T1, T2);
    type Mapper = M;
    fn pack(self) -> TensorTuple<2, Self::Repr, Self::Mapper> {
        TensorTuple::<2, _, _>::pack(self)
    }
}
impl<T1: TensorRepr, T2: TensorRepr, T3: TensorRepr, M: AxisMapper> PackExt<3>
    for (Tensor<T1, M>, Tensor<T2, M>, Tensor<T3, M>)
{
    type Repr = (T1, T2, T3);
    type Mapper = M;
    fn pack(self) -> TensorTuple<3, Self::Repr, Self::Mapper> {
        TensorTuple::<3, _, _>::pack(self)
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
        let (repr, [mapper]) = self.to_tensor().into_raw();
        let mapper = mapper.replace(query)?;
        Ok(unsafe { Tensor::from_raw_unchecked(repr, [mapper]) })
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

impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> ToTensorTuple<N>
    for TensorTuple<N, T, M>
{
    type Repr = T;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self
    }
}
impl<'a, const N: usize, T: AsViewRepr<'a, N>, M: AxisMapper + Clone> ToTensorTuple<N>
    for &'a TensorTuple<N, T, M>
{
    type Repr = T::View;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self.view()
    }
}
impl<'a, const N: usize, T: AsViewMutRepr<'a, N>, M: AxisMapper + Clone> ToTensorTuple<N>
    for &'a mut TensorTuple<N, T, M>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    fn to_tensor_tuple(self) -> TensorTuple<N, Self::Repr, Self::Mapper> {
        self.view_mut()
    }
}

/// Converts a single tensor or tensor reference into a tensor task input.
pub trait ToTensor {
    /// The single-tensor representation type.
    type Repr: TensorRepr;
    /// The axis mapper type.
    type Mapper: AxisMapper;

    /// Converts the input into a single tensor.
    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper>;
}

impl<X: ToTensorTuple<1>> ToTensor for X {
    type Repr = X::Repr;
    type Mapper = X::Mapper;

    fn to_tensor(self) -> Tensor<Self::Repr, Self::Mapper> {
        self.to_tensor_tuple()
    }
}

impl<const N: usize, T: TensorTupleRepr<N> + IsTask, M: AxisMapper> IsTask
    for TensorTuple<N, T, M>
{
}

impl<const N: usize, T: TensorTupleRepr<N> + IsTask, M: AxisMapper, Mk, C: ReprContext<Mk, N, T>>
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

pub unsafe trait TensorTupleContext<
    Mk,
    const N: usize,
    T: TensorTupleRepr<N> + IsTask,
    M: AxisMapper,
>:
    Context<
        Mk,
        TensorTuple<N, T, M>,
        Output = <Self::CType as ContainerImpl<TensorTuple<N, Self::Repr, M>>>::Container,
    >
{
    type Repr: TensorTupleRepr<N>;
    type CType: ContainerImpl<TensorTuple<N, Self::Repr, M>>;
}

unsafe impl<
    const N: usize,
    T: TensorTupleRepr<N> + IsTask,
    M: AxisMapper,
    Mk,
    C: ReprContext<Mk, N, T>,
> TensorTupleContext<Mk, N, T, M> for C
where
    C::CType: ContainerMapImpl<
            <Self as ReprContext<Mk, N, T>>::Repr,
            TensorTuple<N, <Self as ReprContext<Mk, N, T>>::Repr, M>,
        >,
{
    type Repr = C::Repr;
    type CType = C::CType;
}

impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper> IsBindable for TensorTuple<N, T, M> {}

pub type BoundTensorTuple<const N: usize, R, M, RT> = BoundObject<TensorTuple<N, R, M>, RT>;
pub type BoundTensor<R, M, RT> = BoundTensorTuple<1, R, M, RT>;

impl<const N: usize, R: TensorTupleRepr<N>, M: AxisMapper, RT: IsRuntime>
    BoundTensorTuple<N, R, M, RT>
{
    /// Creates an immutable view of every represented tensor.
    pub fn view<'a>(&'a self) -> BoundTensorTuple<N, R::View, M, RT>
    where
        M: Clone,
        R: AsViewRepr<'a, N>,
    {
        BoundTensorTuple::from_raw(self.inner().view(), self.runtime().clone())
    }

    /// Creates a mutable view of every represented tensor.
    pub fn view_mut<'a>(&'a mut self) -> BoundTensorTuple<N, R::ViewMut, M, RT>
    where
        M: Clone,
        R: AsViewMutRepr<'a, N>,
    {
        let runtime = self.runtime().clone();
        BoundTensorTuple::from_raw(self.inner_mut().view_mut(), runtime)
    }

    /// Creates an owned version of every represented tensor.
    pub fn into_owned(self) -> BoundTensorTuple<N, R::Owned, M, RT>
    where
        R: IntoOwnedRepr<N>,
    {
        let (t, rt) = self.into_raw();
        BoundTensorTuple::from_raw(t.into_owned(), rt)
    }
}

/// General utility trait for bound tensor operations.
pub trait BoundTensorExt: ToBoundTensorTuple<1> {
    /// Replace a ID of a leg of the tensor.
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>;
}

impl<T: ToBoundTensorTuple<1>> BoundTensorExt for T {
    fn replace_leg<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<Self::Repr, Self::Mapper, Self::Runtime>,
        <Self::Mapper as ReplaceMapper<Q>>::Err,
    >
    where
        Self::Mapper: ReplaceMapper<Q>,
    {
        let (t, rt) = self.to_bound_tensor_tuple().into_raw();
        t.replace_leg(query)
            .map(|t| BoundTensorTuple::from_raw(t, rt))
    }
}

mod private_to_bound {
    use crate::{
        concept::task::IsRuntime, mapper::AxisMapper, repr::TensorTupleRepr,
        tensor::BoundTensorTuple,
    };
    pub trait ToBoundTensorTupleSealed<const N: usize> {}
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: IsRuntime>
        ToBoundTensorTupleSealed<N> for BoundTensorTuple<N, T, M, RT>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: IsRuntime>
        ToBoundTensorTupleSealed<N> for &BoundTensorTuple<N, T, M, RT>
    {
    }
    impl<const N: usize, T: TensorTupleRepr<N>, M: AxisMapper, RT: IsRuntime>
        ToBoundTensorTupleSealed<N> for &mut BoundTensorTuple<N, T, M, RT>
    {
    }
}

/// Conversion trait to TensorWithRuntime.
///
/// This trait is sealed, and implemented for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime. &TensorWithRuntime and &mut TensorWithRuntime will be converted using view() and view_mut() respectively.
///
/// This trait is useful for unifying the implementation of boilerplate for TensorWithRuntime, &TensorWithRuntime, &mut TensorWithRuntime; this is required because `&` operator is not overloadable.
pub trait ToBoundTensorTuple<const N: usize>:
    private_to_bound::ToBoundTensorTupleSealed<N>
{
    /// The representation type of the resulting tensor.
    type Repr: TensorTupleRepr<N>;
    /// The mapper type of the resulting tensor.
    type Mapper: AxisMapper;
    /// The runtime type of the resulting tensor.
    type Runtime: IsRuntime;
    /// Converts itself to a tensor.
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime>;
}

impl<T: TensorTupleRepr<N>, M: AxisMapper, RT: IsRuntime, const N: usize> ToBoundTensorTuple<N>
    for BoundTensorTuple<N, T, M, RT>
{
    type Repr = T;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        self
    }
}
impl<'a, const N: usize, T: AsViewRepr<'a, N>, M: AxisMapper + Clone, RT: IsRuntime>
    ToBoundTensorTuple<N> for &'a BoundTensorTuple<N, T, M, RT>
{
    type Repr = T::View;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        self.view()
    }
}
impl<'a, const N: usize, T: AsViewMutRepr<'a, N>, M: AxisMapper + Clone, RT: IsRuntime>
    ToBoundTensorTuple<N> for &'a mut BoundTensorTuple<N, T, M, RT>
{
    type Repr = T::ViewMut;
    type Mapper = M;
    type Runtime = RT;
    fn to_bound_tensor_tuple(self) -> BoundTensorTuple<N, Self::Repr, Self::Mapper, Self::Runtime> {
        self.view_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestRepr(usize);

    unsafe impl TensorTupleRepr<1> for TestRepr {
        fn naxes_array(&self) -> [usize; 1] {
            [self.0]
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestMapper(usize);

    unsafe impl AxisMapper for TestMapper {
        type Id = usize;

        fn naxes(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn tensor_tuple_splits_outputs_without_losing_mappers() {
        let outputs = unsafe {
            TensorTuple::from_raw_unchecked(
                (TestRepr(2), TestRepr(3)),
                [TestMapper(2), TestMapper(3)],
            )
        };

        let (first, second) = outputs.unpack();

        assert_eq!(first.repr().0, 2);
        assert_eq!(first.mapper().0, 2);
        assert_eq!(second.repr().0, 3);
        assert_eq!(second.mapper().0, 3);
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct TestRuntime;

    unsafe impl IsRuntime for TestRuntime {}

    #[test]
    fn bound_tensor_tuple_keeps_one_shared_runtime() {
        let tensor = unsafe {
            TensorTuple::from_raw_unchecked(
                (TestRepr(2), TestRepr(2)),
                [TestMapper(2), TestMapper(2)],
            )
        };
        let bound = BoundTensorTuple::from_raw(tensor, TestRuntime);
        let (tensor, runtime) = bound.into_raw();

        assert_eq!(tensor.mapper_array().len(), 2);
        assert_eq!(runtime, TestRuntime);
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
