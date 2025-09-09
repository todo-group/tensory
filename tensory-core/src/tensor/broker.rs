// use alloc::collections::BTreeMap;
// use core::{error, fmt};

// use alloc::vec::Vec;

// /// leg - axis map
// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct LegAlloc<Id: Eq> {
//     legs: Vec<Id>,
// }

use alloc::vec;

use alloc::vec::Vec;
use thiserror::Error;

use crate::tensor::leg::LegMapArg;

pub trait TensorBroker: Sized {
    type Id: Eq + Clone;
    fn len(&self) -> usize;
}

pub trait BuildableBroker<P>: TensorBroker {
    type Err;
    fn build(provider: P) -> Result<Self, Self::Err>;
}

pub unsafe trait OverlayBroker<const N: usize>: TensorBroker {
    type Err;
    fn overlay(brokers: [Self; N]) -> Result<(Self, OverlayAxisOrigin<N>), Self::Err>;
}

pub struct OverlayAxisOrigin<const N: usize> {
    axes: Vec<[usize; N]>,
}
impl<const N: usize> OverlayAxisOrigin<N> {
    pub unsafe fn from_raw_unchecked(raw: Vec<[usize; N]>) -> Self {
        Self { axes: raw }
    }
    pub fn from_raw(raw: Vec<[usize; N]>) -> Result<Self, Vec<[usize; N]>> {
        let n = raw.len();

        let mut seen = vec![false; n];
        for lane in 0..N {
            for i in 0..n {
                seen[i] = false;
            }
            for i in 0..n {
                if raw[i][lane] >= n {
                    return Err(raw);
                }
                if seen[raw[i][lane]] {
                    return Err(raw);
                }
                seen[raw[i][lane]] = true;
            }
        }
        Ok(unsafe { Self::from_raw_unchecked(raw) })
    }
    pub fn len(&self) -> usize {
        self.axes.len()
    }
    pub fn into_raw(self) -> Vec<[usize; N]> {
        self.axes
    }
}

pub unsafe trait ConnectBroker<const N: usize>: TensorBroker {
    type Err;
    fn connect(brokers: [Self; N]) -> Result<(Self, ConnectAxisOrigin<N>), Self::Err>;
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

pub unsafe trait GroupBroker<const N: usize, Q>: TensorBroker {
    type Grouped: GroupedBroker<N, Broker = Self>;
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

// pub unsafe trait SameGroupBroker<const N: usize, Q>: TensorBroker {
//     type Grouped: GroupedBroker<N, Broker = Self>;
//     type Err;
//     fn split(self, queue: Q) -> Result<(Self::Grouped, SameGroupedAxes<N>), Self::Err>;
// }
// pub struct SameGroupedAxes<const N: usize> {
//     len: usize,
//     groups: [Vec<usize>; N],
// }

pub unsafe trait GroupedBroker<const N: usize> {
    type Broker: TensorBroker;
}

pub unsafe trait DecompGroupedBroker<const N: usize, const M: usize>:
    GroupedBroker<N>
{
    type Err;
    fn decomp(
        self,
        conf: DecompConf<N, M, <Self::Broker as TensorBroker>::Id>,
    ) -> Result<[Self::Broker; M], Self::Err>;
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

pub trait TranslateBroker<Content>: TensorBroker {
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
        <Self as TensorBroker>::Id: 'a;
}

#[derive(Error, Debug)]
pub enum DecompError<SE, DE> {
    #[error("Split error: {0}")]
    Split(SE),
    #[error("Decomp error: {0}")]
    Decomp(DE),
}

pub trait ReplaceBroker: TensorBroker {
    type Err;
    fn replace(self, old_leg: &Self::Id, new_leg: Self::Id) -> Result<Self, Self::Err>;
}
