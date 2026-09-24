use alloc::vec::Vec;

use crate::{
    bikeshed::DummyError,
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::{Context, IsRuntime, IsTask, RuntimeErr, RuntimeFor, TaskExt},
    },
    mapper::AxisMapper,
    repr::{TensorRepr, TensorTupleRepr},
    tensor::{
        BoundTensor, BoundTensorTuple, Tensor, TensorTuple, TensorTupleContext, ToBoundTensorTuple,
        ToTensorTuple,
    },
};

/// Overlays several mappers into one mapper and records the index mapping.
///
/// # Safety
///
/// Implementations must preserve the axis structure and ensure that the returned
/// mapping agrees with the returned mapper.
pub unsafe trait OverlayMapper<const N: usize>: AxisMapper {
    /// Error returned when the mappers cannot be overlaid.
    type Err;
    /// Combines `mappers` and returns the corresponding axis mapping.
    unsafe fn overlay(mappers: [Self; N]) -> Result<(Self, AxisAllocation<N>), Self::Err>;
}

/// Maps each axis of an overlaid mapper to an input mapper axis.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AxisAllocation<const N: usize> {
    n: usize,
    maps: [Vec<usize>; N],
}

unsafe impl<const N: usize> RefinedFrom<(usize, [Vec<usize>; N])> for AxisAllocation<N> {
    type Err = DummyError;

    fn verify(&(n, ref maps): &(usize, [Vec<usize>; N])) -> Result<(), Self::Err> {
        let mut seen = alloc::vec![false; n];
        for lane in 0..N {
            for i in seen.iter_mut().take(n) {
                *i = false;
            }
            for i in 0..n {
                if maps[lane][i] >= n {
                    return Err(DummyError);
                }
                if seen[maps[lane][i]] {
                    return Err(DummyError);
                }
                seen[maps[lane][i]] = true;
            }
        }
        Ok(())
    }

    unsafe fn from_raw_unchecked((n, maps): (usize, [Vec<usize>; N])) -> Self {
        Self { n, maps }
    }
    fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.n, self.maps)
    }
}

impl<const N: usize> AxisAllocation<N> {
    /// Returns the number of axes in each input mapper.
    pub fn naxes(&self) -> usize {
        self.n
    }
    pub fn id(n: usize) -> Self {
        let maps = alloc::vec![0..n]
            .into_iter()
            .map(|v| v.collect())
            .collect::<Vec<Vec<usize>>>();
        let maps: [Vec<usize>; N] = maps
            .try_into()
            .unwrap_or_else(|_| panic!("Failed to create identity AxisAllocation"));
        unsafe { Self::from_raw_unchecked((n, maps)) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct UnaryEwiseRepr<const N: usize, T1: TensorTupleRepr<N>, Op> {
    repr1: T1,
    axis_mappings: [AxisAllocation<1>; N],
    op: Op,
}

unsafe impl<const N: usize, T1: TensorTupleRepr<N>, Op>
    RefinedFrom<(T1, [AxisAllocation<1>; N], Op)> for UnaryEwiseRepr<N, T1, Op>
{
    type Err = DummyError;

    fn verify(
        (repr1, axis_mappings, _op): &(T1, [AxisAllocation<1>; N], Op),
    ) -> Result<(), Self::Err> {
        let naxes_array = axis_mappings.each_ref().map(|m| m.naxes());
        if repr1.naxes_array() == naxes_array {
            Ok(())
        } else {
            Err(DummyError)
        }
    }

    unsafe fn from_raw_unchecked(
        (repr1, axis_mappings, op): (T1, [AxisAllocation<1>; N], Op),
    ) -> Self {
        Self {
            repr1,
            axis_mappings,
            op,
        }
    }
    fn into_raw(self) -> (T1, [AxisAllocation<1>; N], Op) {
        (self.repr1, self.axis_mappings, self.op)
    }
}

unsafe impl<const N: usize, T1: TensorTupleRepr<N>, Op> TensorTupleRepr<N>
    for UnaryEwiseRepr<N, T1, Op>
{
    fn naxes_array(&self) -> [usize; N] {
        self.axis_mappings.each_ref().map(|m| m.naxes())
    }
}

impl<const N: usize, T1: TensorTupleRepr<N>, Op> IsTask for UnaryEwiseRepr<N, T1, Op> {}

/// Lazy representation for an addition operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinaryEwiseRepr<const N: usize, T1: TensorTupleRepr<N>, T2: TensorTupleRepr<N>, Op> {
    repr1: T1,
    repr2: T2,
    axis_mappings: [AxisAllocation<2>; N],
    op: Op,
}

unsafe impl<const N: usize, T1: TensorTupleRepr<N>, T2: TensorTupleRepr<N>, Op>
    RefinedFrom<(T1, T2, [AxisAllocation<2>; N], Op)> for BinaryEwiseRepr<N, T1, T2, Op>
{
    type Err = DummyError;

    fn verify(
        (repr1, repr2, axis_mappings, _op): &(T1, T2, [AxisAllocation<2>; N], Op),
    ) -> Result<(), Self::Err> {
        let naxes_array = axis_mappings.each_ref().map(|m| m.naxes());
        if repr1.naxes_array() == naxes_array && repr2.naxes_array() == naxes_array {
            Ok(())
        } else {
            Err(DummyError)
        }
    }

    unsafe fn from_raw_unchecked(
        (repr1, repr2, axis_mappings, op): (T1, T2, [AxisAllocation<2>; N], Op),
    ) -> Self {
        Self {
            repr1,
            repr2,
            axis_mappings,
            op,
        }
    }
    fn into_raw(self) -> (T1, T2, [AxisAllocation<2>; N], Op) {
        (self.repr1, self.repr2, self.axis_mappings, self.op)
    }
}

unsafe impl<const N: usize, T1: TensorTupleRepr<N>, T2: TensorTupleRepr<N>, Op> TensorTupleRepr<N>
    for BinaryEwiseRepr<N, T1, T2, Op>
{
    fn naxes_array(&self) -> [usize; N] {
        self.axis_mappings.each_ref().map(|m| m.naxes())
    }
}

impl<const N: usize, T1: TensorTupleRepr<N>, T2: TensorTupleRepr<N>, Op> IsTask
    for BinaryEwiseRepr<N, T1, T2, Op>
{
}

pub trait EwiseExt<const N: usize>: ToTensorTuple<N> {
    fn ewise1<Op>(self, op: Op) -> TensorTuple<N, UnaryEwiseRepr<N, Self::Repr, Op>, Self::Mapper>
    where
        Self: Sized,
    {
        let (repr1, mappers1) = self.to_tensor_tuple().into_raw();
        let mappers = mappers1;
        let axis_mappings = mappers.each_ref().map(|m| AxisAllocation::id(m.naxes()));
        let repr = unsafe { UnaryEwiseRepr::from_raw_unchecked((repr1, axis_mappings, op)) };
        unsafe { TensorTuple::from_raw_unchecked(repr, mappers) }
    }

    fn ewise2<Op, R>(
        self,
        rhs: R,
        op: Op,
    ) -> Result<
        TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
        <Self::Mapper as OverlayMapper<2>>::Err,
    >
    where
        R: ToTensorTuple<N, Mapper = Self::Mapper>,
        Self: Sized,
        Self::Mapper: OverlayMapper<2>,
    {
        let (repr1, mappers1) = self.to_tensor_tuple().into_raw();
        let (repr2, mappers2) = rhs.to_tensor_tuple().into_raw();
        let (mappers, axis_mappings): (Vec<_>, Vec<_>) = mappers1
            .into_iter()
            .zip(mappers2)
            .map(|(m1, m2)| unsafe { Self::Mapper::overlay([m1, m2]) })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let mappers = mappers.try_into().map_err(|_| DummyError).unwrap();
        let axis_mappings = axis_mappings.try_into().unwrap();
        let repr =
            unsafe { BinaryEwiseRepr::from_raw_unchecked((repr1, repr2, axis_mappings, op)) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, mappers) })
    }
}
impl<const N: usize, X: ToTensorTuple<N>> EwiseExt<N> for X {}

// pub trait BoundEwiseExt<const N: usize>: ToBoundTensorTuple<N> {
//     fn ewise2<Op, R, Err>(
//         self,
//         rhs: R,
//         op: Op,
//     ) -> Result<
//         BoundTensorTuple<
//             N,
//             <<Self::Runtime as RuntimeFor<
//                 TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
//             >>::Ctx as TensorTupleContext<
//                 <Self::Runtime as RuntimeFor<
//                     TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
//                 >>::Mk,
//                 2,
//                 BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>,
//                 Self::Mapper,
//             >>::Repr,
//             Self::Mapper,
//             Self::Runtime,
//         >,
//         RuntimeErr<<Self::Mapper as OverlayMapper<N>>::Err, Err>,
//     >
//     where
//         Self: Sized,
//         R: ToBoundTensorTuple<N, Mapper = Self::Mapper, Runtime = Self::Runtime>,
//         Self::Mapper: OverlayMapper<N>,
//         Self::Runtime:
//             RuntimeFor<TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>>,
//         <Self::Runtime as RuntimeFor<
//             TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
//         >>::Ctx: TensorTupleContext<
//                 <Self::Runtime as RuntimeFor<
//                     TensorTuple<N, BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
//                 >>::Mk,
//                 N,
//                 BinaryEwiseRepr<N, Self::Repr, R::Repr, Op>,
//                 Self::Mapper,
//                 CType = Resulting<Raw, Err>,
//             >,
//     {
//         let (t1, rt1) = self.to_bound_tensor_tuple().into_raw();
//         let (t2, rt2) = rhs.to_bound_tensor_tuple().into_raw();
//         if rt1 != rt2 {
//             return Err(RuntimeErr::Runtime);
//         }
//         let rt = rt1;

//         let task = t1.ewise2(t2, op).map_err(RuntimeErr::Defer)?;
//         let res = task.with(rt.ctx()).map_err(RuntimeErr::Execute)?;
//         Ok(res.bind(rt))
//     }
// }
// impl<const N: usize, X: ToBoundTensorTuple<N>> BoundEwiseExt<N> for X {}
