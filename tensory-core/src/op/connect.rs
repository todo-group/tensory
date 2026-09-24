use alloc::vec::Vec;

use crate::{
    bikeshed::DummyError,
    concept::{refined::RefinedFrom, task::IsTask},
    mapper::AxisMapper,
    repr::TensorTupleRepr,
    tensor::{TensorTuple, ToTensorTuple},
};

/// Connects axes from several mappers and records the connected origins.
///
/// # Safety
///
/// Implementations must preserve the axis structure and ensure that every
/// reported connection is represented by the returned mapper.
pub unsafe trait ConnectMapper<const N: usize>: AxisMapper {
    /// Error returned when the mappers cannot be connected.
    type Err;
    /// Connects `mappers` and returns the connection metadata.
    unsafe fn connect(mappers: [Self; N]) -> Result<(Self, AxisConnection<N>), Self::Err>;
}

/// Describes which input axes were connected.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AxisConnection<const N: usize> {
    in_lens: [usize; N],
    connections: Vec<((usize, usize), (usize, usize))>,
}

unsafe impl<const N: usize> RefinedFrom<([usize; N], Vec<((usize, usize), (usize, usize))>)>
    for AxisConnection<N>
{
    type Err = DummyError;

    unsafe fn from_raw_unchecked(
        (in_lens, connections): ([usize; N], Vec<((usize, usize), (usize, usize))>),
    ) -> Self {
        Self {
            in_lens,
            connections,
        }
    }
    /// Creates validated connection metadata.
    fn verify(
        &(in_lens, ref connections): &([usize; N], Vec<((usize, usize), (usize, usize))>),
    ) -> Result<(), Self::Err> {
        let mut rec: Vec<_> = in_lens.iter().map(|&x| alloc::vec![false; x]).collect();
        for &((t1, i1), (t2, i2)) in connections.iter() {
            if t1 >= N || i1 >= in_lens[t1] || t2 >= N || i2 >= in_lens[t2] || t1 == t2 {
                return Err(DummyError);
            }
            if rec[t1][i1] || rec[t2][i2] {
                return Err(DummyError);
            }
            rec[t1][i1] = true;
            rec[t2][i2] = true;
        }
        for r in rec {
            for v in r {
                if !v {
                    return Err(DummyError);
                }
            }
        }
        Ok(())
    }
    /// Decomposes the metadata into its raw components.
    fn into_raw(self) -> ([usize; N], Vec<((usize, usize), (usize, usize))>) {
        (self.in_lens, self.connections)
    }
}

impl<const N: usize> AxisConnection<N> {
    /// Returns the input axis lengths.
    pub fn in_lens(&self) -> [usize; N] {
        self.in_lens
    }
    /// Returns the number of axes remaining after connections.
    pub fn len(&self) -> usize {
        self.in_lens.iter().sum::<usize>() - 2 * self.connections.len()
    }
    /// Returns whether no axes remain after connections.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Lazy representation for a tensor contraction operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BinaryConnectRepr<const N: usize, L: TensorTupleRepr<N>, R: TensorTupleRepr<N>, Op> {
    lhs: L,
    rhs: R,
    axis_connections: [AxisConnection<2>; N],
    op: Op,
}

unsafe impl<const N: usize, L: TensorTupleRepr<N>, R: TensorTupleRepr<N>, Op>
    RefinedFrom<(L, R, [AxisConnection<2>; N], Op)> for BinaryConnectRepr<N, L, R, Op>
{
    type Err = DummyError;

    fn verify(
        (lhs, rhs, axis_connections, _op): &(L, R, [AxisConnection<2>; N], Op),
    ) -> Result<(), Self::Err> {
        let naxes_pair_array = axis_connections.each_ref().map(|m| m.in_lens());
        let lhs_naxes_array = naxes_pair_array.map(|[lhs, _rhs]| lhs);
        let rhs_naxes_array = naxes_pair_array.map(|[_lhs, rhs]| rhs);
        if lhs_naxes_array != lhs.naxes_array() && rhs_naxes_array != rhs.naxes_array() {
            return Err(DummyError);
        }
        Ok(())
    }

    unsafe fn from_raw_unchecked(
        (lhs, rhs, axis_connections, op): (L, R, [AxisConnection<2>; N], Op),
    ) -> Self {
        Self {
            lhs,
            rhs,
            axis_connections,
            op,
        }
    }

    fn into_raw(self) -> (L, R, [AxisConnection<2>; N], Op) {
        (self.lhs, self.rhs, self.axis_connections, self.op)
    }
}

unsafe impl<const N: usize, L: TensorTupleRepr<N>, R: TensorTupleRepr<N>, Op> TensorTupleRepr<N>
    for BinaryConnectRepr<N, L, R, Op>
{
    fn naxes_array(&self) -> [usize; N] {
        self.axis_connections.each_ref().map(|origin| origin.len())
    }
}

impl<const N: usize, L: TensorTupleRepr<N>, R: TensorTupleRepr<N>, Op> IsTask
    for BinaryConnectRepr<N, L, R, Op>
{
}

pub trait ConnectExt<const N: usize>: ToTensorTuple<N> {
    fn connect<Op, R>(
        self,
        rhs: R,
        op: Op,
    ) -> Result<
        TensorTuple<N, BinaryConnectRepr<N, Self::Repr, R::Repr, Op>, Self::Mapper>,
        <Self::Mapper as ConnectMapper<2>>::Err,
    >
    where
        R: ToTensorTuple<N, Mapper = Self::Mapper>,
        Self: Sized,
        Self::Mapper: ConnectMapper<2>,
    {
        let (repr1, mappers1) = self.to_tensor_tuple().into_raw();
        let (repr2, mappers2) = rhs.to_tensor_tuple().into_raw();
        let (mappers, axis_connectings): (Vec<_>, Vec<_>) = mappers1
            .into_iter()
            .zip(mappers2)
            .map(|(m1, m2)| unsafe { Self::Mapper::connect([m1, m2]) })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let mappers = mappers.try_into().map_err(|_| DummyError).unwrap();
        let axis_connectings = axis_connectings.try_into().unwrap();
        let repr =
            unsafe { BinaryConnectRepr::from_raw_unchecked((repr1, repr2, axis_connectings, op)) };
        Ok(unsafe { TensorTuple::from_raw_unchecked(repr, mappers) })
    }
}

impl<const N: usize, T: ToTensorTuple<N>> ConnectExt<N> for T {}
