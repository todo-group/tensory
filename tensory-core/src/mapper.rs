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

/// Constructs a mapper from an owned precursor value.
pub trait BuildableMapper<P>: AxisMapper {
    /// Error returned when the precursor cannot form a valid mapper.
    type Err;
    /// Builds a mapper from `precursor`.
    fn build(precursor: P) -> Result<Self, Self::Err>;
}

/// Constructs a mapper from a precursor while preserving its source data.
pub trait SynBuildableMapper<P>: AxisMapper {
    /// Error returned when the precursor cannot form a valid mapper.
    type Err;
    /// Builds a mapper from `precursor` without consuming the source semantics.
    fn syn_build(precursor: P) -> Result<Self, Self::Err>;
}

/// Groups axes from a mapper according to a queue.
///
/// # Safety
///
/// Implementations must preserve the mapper's axis structure and ensure that
/// the returned groups describe exactly the input axes.
pub unsafe trait GroupMapper<const N: usize, Q>: AxisMapper {
    /// Mapper produced by the grouping operation.
    type Grouped: GroupedMapper<N, Mapper = Self>;
    /// Error returned when grouping fails.
    type Err;
    /// Groups the mapper's axes according to `queue`.
    fn split(self, queue: Q) -> Result<(Self::Grouped, GroupedAxes<N>), Self::Err>;
}
/// Records the groups produced by a grouping operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct GroupedAxes<const N: usize> {
    len: usize,
    groups: [Vec<usize>; N],
}
impl<const N: usize> GroupedAxes<N> {
    /// Creates group metadata without validating its indices.
    ///
    /// # Safety
    ///
    /// Every input axis in `0..len` must occur exactly once in `groups`.
    pub unsafe fn from_raw_unchecked(len: usize, groups: [Vec<usize>; N]) -> Self {
        Self { len, groups }
    }
    /// Creates validated group metadata.
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
        if seen.iter().take(len).any(|present| !present) {
            return Err((len, groups));
        }
        Ok(unsafe { Self::from_raw_unchecked(len, groups) })
    }
    /// Returns the number of input axes represented by the groups.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns whether the groups represent no input axes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Decomposes the metadata into its raw components.
    pub fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.len, self.groups)
    }
}

/// Splits axes into equally sized groups.
///
/// # Safety
///
/// Implementations must preserve the mapper's axis structure and ensure that
/// the returned groups are complete and have equal lengths.
pub unsafe trait EquivGroupMapper<const N: usize, Q>: AxisMapper {
    /// Mapper produced by the grouping operation.
    type Grouped: GroupedMapper<N, Mapper = Self>;
    /// Error returned when grouping fails.
    type Err;
    /// Splits the mapper's axes into equivalent groups.
    fn equiv_split(self, queue: Q) -> Result<(Self::Grouped, EquivGroupedAxes<N>), Self::Err>;
}
/// Records equally sized groups produced by a grouping operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct EquivGroupedAxes<const N: usize> {
    len: usize,
    groups: [Vec<usize>; N],
}
impl<const N: usize> EquivGroupedAxes<N> {
    /// Creates group metadata without validating its indices.
    ///
    /// # Safety
    ///
    /// Every input axis in `0..len` must occur exactly once in `groups`, and
    /// every group must have the same length.
    pub unsafe fn from_raw_unchecked(len: usize, groups: [Vec<usize>; N]) -> Self {
        Self { len, groups }
    }
    /// Creates validated equivalent-group metadata.
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
        if seen.iter().take(len).any(|present| !present) {
            return Err((len, groups));
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
    /// Returns the number of input axes represented by the groups.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Returns whether the groups represent no input axes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Decomposes the metadata into its raw components.
    pub fn into_raw(self) -> (usize, [Vec<usize>; N]) {
        (self.len, self.groups)
    }
}

/// Provides access to the mapper produced by a grouping operation.
///
/// # Safety
///
/// Implementations must ensure that the associated mapper remains consistent
/// with the grouping metadata.
pub unsafe trait GroupedMapper<const N: usize> {
    /// Mapper associated with the grouped axes.
    type Mapper: AxisMapper;
}

/// Decomposes grouped axes into a fixed number of output mappers.
///
/// # Safety
///
/// Implementations must preserve the grouped mapper's axis semantics and make
/// the returned mappers agree with `conf`.
pub unsafe trait DecompGroupedMapper<const N: usize, const M: usize>:
    GroupedMapper<N>
{
    /// Error returned when decomposition fails.
    type Err;
    /// Decomposes the grouped mapper according to `conf`.
    fn decomp(
        self,
        conf: DecompConf<N, M, <Self::Mapper as AxisMapper>::Id>,
    ) -> Result<[Self::Mapper; M], Self::Err>;
}

/// Describes how grouped axes should be decomposed.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DecompConf<const N: usize, const M: usize, Id> {
    group_belongs: [usize; N],
    new_bonds: Vec<((usize, Id), (usize, Id))>,
}
impl<const N: usize, const M: usize, Id> DecompConf<N, M, Id> {
    /// Creates a decomposition configuration without validation.
    ///
    /// # Safety
    ///
    /// Every group index must be less than `M`, and connected groups must be
    /// distinct.
    pub unsafe fn from_raw_unchecked(
        group_belongs: [usize; N],
        new_bonds: Vec<((usize, Id), (usize, Id))>,
    ) -> Self {
        Self {
            group_belongs,
            new_bonds,
        }
    }
    /// Creates a validated decomposition configuration.
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
    /// Decomposes the configuration into its raw components.
    pub fn into_raw(self) -> ([usize; N], Vec<((usize, Id), (usize, Id))>) {
        (self.group_belongs, self.new_bonds)
    }
}

/// Solves grouped axes into a fixed number of output mappers.
///
/// # Safety
///
/// Implementations must preserve the grouped mapper's axis semantics and make
/// the returned mappers agree with `conf`.
pub unsafe trait SolveGroupedMapper<const N: usize, const M: usize>:
    GroupedMapper<N>
{
    /// Error returned when solving fails.
    type Err;
    /// Solves the grouped mapper according to `conf`.
    fn solve(
        self,
        conf: SolveConf<N, M, <Self::Mapper as AxisMapper>::Id>,
    ) -> Result<[Self::Mapper; M], Self::Err>;
}

/// Describes how grouped axes are assigned to output mappers.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SolveConf<const N: usize, const M: usize, Id> {
    group_belongs: [[bool; N]; M],
    new_legs: Vec<(usize, Id)>,
}
impl<const N: usize, const M: usize, Id> SolveConf<N, M, Id> {
    /// Creates a solve configuration without validation.
    ///
    /// # Safety
    ///
    /// Every output index in `new_legs` must be less than `M`.
    pub unsafe fn from_raw_unchecked(
        group_belongs: [[bool; N]; M],
        new_legs: Vec<(usize, Id)>,
    ) -> Self {
        Self {
            group_belongs,
            new_legs,
        }
    }
    /// Creates a validated solve configuration.
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
    /// Decomposes the configuration into its raw components.
    pub fn into_raw(self) -> ([[bool; N]; M], Vec<(usize, Id)>) {
        (self.group_belongs, self.new_legs)
    }
}

/// Reorders values according to the mapper's axis IDs.
pub trait SortMapper<Content>: AxisMapper {
    /// Error returned when an axis cannot be translated.
    type Err;
    /// Sorts `map` into the mapper's axis order.
    fn sort<
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

/// Error returned by a split followed by a decomposition operation.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Error)]
pub enum SplittyErr<SE, DE> {
    /// The initial axis split failed.
    #[error("Split error: {0}")]
    Split(SE),
    /// The operation using the split result failed.
    #[error("Use error: {0}")]
    Use(DE),
}

/// Replaces the mapper's selected axes.
pub trait ReplaceMapper<Q>: AxisMapper {
    /// Error returned when replacement fails.
    type Err;
    /// Replaces axes selected by `query`.
    fn replace(self, query: Q) -> Result<Self, Self::Err>;
}

/// Translates an external axis identifier into another representation.
pub trait TranslateMapper<Exp>: AxisMapper {
    /// Result type produced by translation.
    type Res;
    /// Error returned when translation fails.
    type Err;
    /// Translates `leg`.
    fn translate(&self, leg: Exp) -> Result<Self::Res, Self::Err>;
}
