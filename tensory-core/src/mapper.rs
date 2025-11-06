//! Mapper concept: translator between layer 1 and 2 axis descriptions.

use alloc::vec;

use alloc::vec::Vec;
use thiserror::Error;

use crate::args::LegMapArg;

/// Minimal interface for tensor axis mappers.
///
/// In the logical model, a mapper is a injective mapping from usize indices `0`, `1`, ..., `naxes()-1`, each representing an axis of a tensor, to unique ID indices.
///
/// In the conceptual model, a mapper `m` is sound if and only if it is bound with a tensor repr `a` satisfying `m.naxes() == a.naxes()`, and the compound behaves as a tensor with axes indexed by locally-unique IDs. We refer the axes indexed with IDs, as described before, as "legs". See `Tensor` for more integrated explanation of the concept of legs.
///
/// In practice, a type implementing this trait serves as a opaque translator between the usize indices and the ID indices. The actual mapping is not exposed in this trait, and the core functions are defined as sub-traits e.g. `OverlayMapper`.
///
/// # Safety
///
/// The implementor MUST ensure the following invariants:
///
/// - The number of axes (=: naxes()) is same with the number of axes of the bound tensor representation, even through mutable operations. (this also means the number of axes is fixed for the same object)
/// - The mapping from usize indices and ID indices are never changed for the same object, even through mutable operations.
///
/// We refer the above invariants AND axis structure together as "semantic structure of legs" or simply "leg structure".
///
/// `mem::{swap,replace,take,...}` syntactically violate the above conditons, but these operations semantically do not change the objects but move them. So we think the above conditions are not violated by these operations.
///
/// Implicitly (by definition), the implementor MUST ensure the following condition:
///
/// - The ID is locally unique; for the same mapper object, no two axes are mapped to the same ID.
///
/// # Note
///
/// The implementor MAY implement mapping modification operations (e.g. `replace`,`swap`,`remap`) as by-value methods. `Tensor` provides hatches to use by-value modifications.
pub unsafe trait AxisMapper: Sized {
    /// The type of unique IDs used as indices the axes.
    type Id: Eq;
    /// Returns the number of axes of the tensor. this number is fixed for the same object even through mutable operations.
    ///
    /// this function serves as a dynamic version of `const N:usize`.
    fn naxes(&self) -> usize;
}

pub trait BuildableMapper<P>: AxisMapper {
    type Err;
    fn build(precursor: P) -> Result<Self, Self::Err>;
}

pub trait SynBuildableMapper<P>: AxisMapper {
    type Err;
    fn syn_build(precursor: P) -> Result<Self, Self::Err>;
}

pub unsafe trait OverlayMapper<const N: usize>: AxisMapper {
    type Err;
    fn overlay(mappers: [Self; N]) -> Result<(Self, OverlayAxisMapping<N>), Self::Err>;
}

pub struct OverlayAxisMapping<const N: usize> {
    n: usize,
    maps: [Vec<usize>; N],
}
impl<const N: usize> OverlayAxisMapping<N> {
    pub unsafe fn from_raw_unchecked(n: usize, maps: [Vec<usize>; N]) -> Self {
        Self { n, maps }
    }
    pub fn from_raw(n: usize, maps: [Vec<usize>; N]) -> Result<Self, (usize, [Vec<usize>; N])> {
        let mut seen = vec![false; n];
        for lane in 0..N {
            for i in seen.iter_mut().take(n) {
                *i = false;
            }
            for i in 0..n {
                if maps[lane][i] >= n {
                    return Err((n, maps));
                }
                if seen[maps[lane][i]] {
                    return Err((n, maps));
                }
                seen[maps[lane][i]] = true;
            }
        }
        Ok(unsafe { Self::from_raw_unchecked(n, maps) })
    }
    pub fn dim(&self) -> usize {
        self.n
    }
    pub fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.n, self.maps)
    }
}

pub unsafe trait ConnectMapper<const N: usize>: AxisMapper {
    type Err;
    fn connect(mappers: [Self; N]) -> Result<(Self, ConnectAxisOrigin<N>), Self::Err>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectAxisOrigin<const N: usize> {
    in_lens: [usize; N],
    axis_connection: Vec<((usize, usize), (usize, usize))>,
}
impl<const N: usize> ConnectAxisOrigin<N> {
    pub unsafe fn from_raw_unchecked(
        in_lens: [usize; N],
        axis_connection: Vec<((usize, usize), (usize, usize))>,
    ) -> Self {
        Self {
            in_lens,
            axis_connection,
        }
    }
    pub fn from_raw(
        in_lens: [usize; N],
        axis_connection: Vec<((usize, usize), (usize, usize))>,
    ) -> Result<Self, ([usize; N], Vec<((usize, usize), (usize, usize))>)> {
        let mut rec: Vec<_> = in_lens.iter().map(|&x| vec![false; x]).collect();
        for &((t1, i1), (t2, i2)) in axis_connection.iter() {
            if t1 >= N || i1 >= in_lens[t1] || t2 >= N || i2 >= in_lens[t2] || t1 == t2 {
                return Err((in_lens, axis_connection));
            }
            if rec[t1][i1] || rec[t2][i2] {
                return Err((in_lens, axis_connection));
            }
            rec[t1][i1] = true;
            rec[t2][i2] = true;
        }
        for r in rec {
            for v in r {
                if !v {
                    return Err((in_lens, axis_connection));
                }
            }
        }

        Ok(unsafe { Self::from_raw_unchecked(in_lens, axis_connection) })
    }

    pub fn in_lens(&self) -> &[usize] {
        &self.in_lens
    }
    pub fn len(&self) -> usize {
        self.in_lens.iter().sum::<usize>() - 2 * self.axis_connection.len()
    }
    pub fn into_raw(self) -> ([usize; N], Vec<((usize, usize), (usize, usize))>) {
        (self.in_lens, self.axis_connection)
    }
}

pub unsafe trait GroupMapper<const N: usize, Q>: AxisMapper {
    type Grouped: GroupedMapper<N, Mapper = Self>;
    type Err;
    fn split(self, queue: Q) -> Result<(Self::Grouped, GroupedAxes<N>), Self::Err>;
}
pub struct GroupedAxes<const N: usize> {
    len: usize,
    groups: [Vec<usize>; N],
}
impl<const N: usize> GroupedAxes<N> {
    pub unsafe fn from_raw_unchecked(len: usize, groups: [Vec<usize>; N]) -> Self {
        Self { len, groups }
    }
    pub fn from_raw(len: usize, groups: [Vec<usize>; N]) -> Result<Self, (usize, [Vec<usize>; N])> {
        let mut seen = vec![false; len];
        for lane in 0..N {
            for &i in groups[lane].iter() {
                if i >= len || seen[i] {
                    return Err((len, groups));
                }
                seen[i] = true;
            }
        }
        for i in 0..len {
            if !seen[i] {
                return Err((len, groups));
            }
        }
        Ok(unsafe { Self::from_raw_unchecked(len, groups) })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.len, self.groups)
    }
}

pub unsafe trait EquivGroupMapper<const N: usize, Q>: AxisMapper {
    type Grouped: GroupedMapper<N, Mapper = Self>;
    type Err;
    fn equiv_split(self, queue: Q) -> Result<(Self::Grouped, EquivGroupedAxes<N>), Self::Err>;
}
pub struct EquivGroupedAxes<const N: usize> {
    len: usize,
    groups: [Vec<usize>; N],
}
impl<const N: usize> EquivGroupedAxes<N> {
    pub unsafe fn from_raw_unchecked(len: usize, groups: [Vec<usize>; N]) -> Self {
        Self { len, groups }
    }
    pub fn from_raw(len: usize, groups: [Vec<usize>; N]) -> Result<Self, (usize, [Vec<usize>; N])> {
        let mut seen = vec![false; len];
        for lane in 0..N {
            for &i in groups[lane].iter() {
                if i >= len || seen[i] {
                    return Err((len, groups));
                }
                seen[i] = true;
            }
        }
        for i in 0..len {
            if !seen[i] {
                return Err((len, groups));
            }
        }
        if N > 0 {
            let glen = groups[0].len();
            for lane in 0..N {
                if glen != groups[lane].len() {
                    return Err((len, groups));
                }
            }
        }

        Ok(unsafe { Self::from_raw_unchecked(len, groups) })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.len, self.groups)
    }
}

pub unsafe trait GroupedMapper<const N: usize> {
    type Mapper: AxisMapper;
}

pub unsafe trait DecompGroupedMapper<const N: usize, const M: usize>:
    GroupedMapper<N>
{
    type Err;
    fn decomp(
        self,
        conf: DecompConf<N, M, <Self::Mapper as AxisMapper>::Id>,
    ) -> Result<[Self::Mapper; M], Self::Err>;
}
pub struct DecompConf<const N: usize, const M: usize, Id> {
    group_belongs: [usize; N],
    new_bonds: Vec<((usize, Id), (usize, Id))>,
}
impl<const N: usize, const M: usize, Id> DecompConf<N, M, Id> {
    pub unsafe fn from_raw_unchecked(
        group_belongs: [usize; N],
        new_bonds: Vec<((usize, Id), (usize, Id))>,
    ) -> Self {
        Self {
            group_belongs,
            new_bonds,
        }
    }
    pub fn from_raw(
        group_belongs: [usize; N],
        new_bonds: Vec<((usize, Id), (usize, Id))>,
    ) -> Result<Self, ([usize; N], Vec<((usize, Id), (usize, Id))>)> {
        for &g in group_belongs.iter() {
            if g >= M {
                return Err((group_belongs, new_bonds));
            }
        }
        for &((g1, _), (g2, _)) in new_bonds.iter() {
            if g1 >= M || g2 >= M || g1 == g2 {
                return Err((group_belongs, new_bonds));
            }
        }
        Ok(unsafe { Self::from_raw_unchecked(group_belongs, new_bonds) })
    }
    pub fn into_raw(self) -> ([usize; N], Vec<((usize, Id), (usize, Id))>) {
        (self.group_belongs, self.new_bonds)
    }
}

pub unsafe trait SolveGroupedMapper<const N: usize, const M: usize>:
    GroupedMapper<N>
{
    type Err;
    fn solve(
        self,
        conf: SolveConf<N, M, <Self::Mapper as AxisMapper>::Id>,
    ) -> Result<[Self::Mapper; M], Self::Err>;
}
pub struct SolveConf<const N: usize, const M: usize, Id> {
    group_belongs: [[bool; N]; M],
    new_legs: Vec<(usize, Id)>,
}
impl<const N: usize, const M: usize, Id> SolveConf<N, M, Id> {
    pub unsafe fn from_raw_unchecked(
        group_belongs: [[bool; N]; M],
        new_legs: Vec<(usize, Id)>,
    ) -> Self {
        Self {
            group_belongs,
            new_legs,
        }
    }
    pub fn from_raw(
        group_belongs: [[bool; N]; M],
        new_legs: Vec<(usize, Id)>,
    ) -> Result<Self, ([[bool; N]; M], Vec<(usize, Id)>)> {
        for &(g, _) in new_legs.iter() {
            if g >= M {
                return Err((group_belongs, new_legs));
            }
        }
        Ok(unsafe { Self::from_raw_unchecked(group_belongs, new_legs) })
    }
    pub fn into_raw(self) -> ([[bool; N]; M], Vec<(usize, Id)>) {
        (self.group_belongs, self.new_legs)
    }
}

pub trait TranslateMapper<Content>: AxisMapper {
    type Err;
    fn translate<
        'a,
        K: ExactSizeIterator + Iterator<Item = &'a Self::Id>,
        V: ExactSizeIterator + Iterator<Item = Content>,
    >(
        &self,
        map: LegMapArg<K, V>,
    ) -> Result<Vec<Content>, Self::Err>
    where
        <Self as AxisMapper>::Id: 'a;
}

#[derive(Error, Debug)]
pub enum SplittyError<SE, DE> {
    #[error("Split error: {0}")]
    Split(SE),
    #[error("Use error: {0}")]
    Use(DE),
}

pub trait ReplaceMapper<Q>: AxisMapper {
    type Err;
    fn replace(self, query: Q) -> Result<Self, Self::Err>;
}
