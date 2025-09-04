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

pub trait TensorBroker: Sized {
    type Id: Eq + Clone;
    fn len(&self) -> usize;
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
    axis_origin: Vec<(usize, usize)>,
    axis_connection: Vec<((usize, usize), (usize, usize))>,
}
impl<const N: usize> ConnectAxisOrigin<N> {
    pub unsafe fn from_raw_unchecked(
        in_lens: [usize; N],
        axis_origin: Vec<(usize, usize)>,
        axis_connection: Vec<((usize, usize), (usize, usize))>,
    ) -> Self {
        Self {
            in_lens,
            axis_origin,
            axis_connection,
        }
    }
    pub fn from_raw(
        in_lens: [usize; N],
        axis_origin: Vec<(usize, usize)>,
        axis_connection: Vec<((usize, usize), (usize, usize))>,
    ) -> Result<
        Self,
        (
            [usize; N],
            Vec<(usize, usize)>,
            Vec<((usize, usize), (usize, usize))>,
        ),
    > {
        let mut rec: Vec<_> = in_lens.iter().map(|&x| vec![false; x]).collect();
        for &(t, i) in axis_origin.iter() {
            if t >= N || i >= in_lens[t] {
                return Err((in_lens, axis_origin, axis_connection));
            }
            if rec[t][i] {
                return Err((in_lens, axis_origin, axis_connection));
            }
            rec[t][i] = true;
        }
        for &((t1, i1), (t2, i2)) in axis_connection.iter() {
            if t1 >= N || i1 >= in_lens[t1] || t2 >= N || i2 >= in_lens[t2] || t1 == t2 {
                return Err((in_lens, axis_origin, axis_connection));
            }
            if rec[t1][i1] || rec[t2][i2] {
                return Err((in_lens, axis_origin, axis_connection));
            }
            rec[t1][i1] = true;
            rec[t2][i2] = true;
        }
        for r in rec {
            for v in r {
                if !v {
                    return Err((in_lens, axis_origin, axis_connection));
                }
            }
        }

        Ok(unsafe { Self::from_raw_unchecked(in_lens, axis_origin, axis_connection) })
    }

    pub fn in_lens(&self) -> &[usize] {
        &self.in_lens
    }
    pub fn len(&self) -> usize {
        self.axis_origin.len()
    }
    pub fn into_raw(
        self,
    ) -> (
        [usize; N],
        Vec<(usize, usize)>,
        Vec<((usize, usize), (usize, usize))>,
    ) {
        (self.in_lens, self.axis_origin, self.axis_connection)
    }
}

pub unsafe trait DecompBroker: TensorBroker {
    type PreManager;
    type PreBond;
    type Err;
    fn decomp<const N: usize>(
        self,
        pre_managers: [Self::PreManager; N],
        pre_bonds: Vec<(usize, usize, Self::PreBond)>,
    ) -> Result<([Self; N], [Vec<DecompAxisOrigin>; N]), Self::Err>;
}
pub enum DecompAxisOrigin {
    Inherited(usize),
    Introduced(usize),
}

pub trait TranslateMgr<M>: TensorBroker {
    type Content;
    type Err;
    fn translate(&self, indices: M) -> Result<Vec<Self::Content>, Self::Err>;
}

// pub trait ShareMgr {
//     fn
// }
