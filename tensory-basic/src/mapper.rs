//impl<Id: LegId> LegId for Prime<Id> {}

// fn vec_to_maybe_uninit_vec<T>(v: Vec<T>) -> Vec<MaybeUninit<T>> {
//     let mut v = ManuallyDrop::new(v);
//     let p = v.as_mut_ptr();
//     let len = v.len();
//     let cap = v.capacity();
//     unsafe { Vec::from_raw_parts(p as _, len, cap) }
// }

use core::{convert::Infallible, error::Error, fmt::Display};
use thiserror::Error;

use alloc::vec::Vec;

use tensory_core::{
    args::{LegMapArg, LegSetArg},
    mapper::{
        AxisMapper, BuildableMapper, ConnectAxisOrigin, ConnectMapper, DecompConf,
        DecompGroupedMapper, GroupMapper, GroupedAxes, GroupedMapper, OverlayAxisMapping,
        OverlayMapper, ReplaceMapper, TranslateMapper,
    },
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecMapper<T>(Vec<T>);
impl<T> VecMapper<T> {
    pub unsafe fn from_raw_unchecked(raw: Vec<T>) -> Self {
        VecMapper(raw)
    }
    pub fn from_raw(raw: Vec<T>) -> Result<Self, Vec<T>>
    where
        T: Eq,
    {
        if !check_unique(&raw) {
            return Err(raw);
        }
        Ok(unsafe { Self::from_raw_unchecked(raw) })
    }
}

unsafe impl<T: Eq + Clone> AxisMapper for VecMapper<T> {
    type Id = T;

    fn naxes(&self) -> usize {
        self.0.len()
    }
}

fn check_unique<Id: Eq>(legs: &[Id]) -> bool {
    for i in 0..legs.len() {
        for j in (i + 1)..legs.len() {
            if legs[i] == legs[j] {
                return false;
            }
        }
    }
    true
}

#[derive(Debug)]
pub struct BuildErr;
impl Display for BuildErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "build error")
    }
}
impl Error for BuildErr {}

impl<T: Eq + Clone, I: Iterator<Item = Self::Id>> BuildableMapper<I> for VecMapper<T> {
    type Err = BuildErr;
    fn build(iter: I) -> Result<Self, Self::Err> {
        let v = iter.collect();
        Self::from_raw(v).map_err(|_| BuildErr)
    }
}

#[derive(Debug)]
pub struct OverlayErr;
impl Display for OverlayErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "overlay error")
    }
}
impl Error for OverlayErr {}

unsafe impl<T: Eq + Clone> OverlayMapper<2> for VecMapper<T> {
    type Err = OverlayErr;

    fn overlay([lhs, rhs]: [Self; 2]) -> Result<(Self, OverlayAxisMapping<2>), Self::Err> {
        let len = lhs.naxes();
        let rhs_len = rhs.naxes();
        if len != rhs_len {
            return Err(OverlayErr);
        }

        let rhs_vec: Option<Vec<usize>> = lhs
            .0
            .iter()
            .map(|id| rhs.0.iter().position(|e| e == id))
            .collect();
        let rhs_vec = rhs_vec.ok_or(OverlayErr)?;

        OverlayAxisMapping::from_raw(len, [(0..len).collect(), rhs_vec])
            .map_err(|_| OverlayErr)
            .map(|m| (lhs, m))
    }
}

unsafe impl<T: Eq + Clone> ConnectMapper<2> for VecMapper<T> {
    type Err = Infallible;
    fn connect([lhs, rhs]: [Self; 2]) -> Result<(Self, ConnectAxisOrigin<2>), Self::Err> {
        let lhs_len = lhs.naxes();
        let rhs_len = rhs.naxes();

        let pairs: Vec<_> = lhs
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, l_leg)| {
                rhs.0
                    .iter()
                    .enumerate()
                    .find(|(_, r_leg)| l_leg == *r_leg)
                    .map(|(j, _)| ((0, i), (1, j)))
            })
            .collect();

        let mut lhs_rest = lhs.0.into_iter().map(|e| Some(e)).collect::<Vec<_>>();
        let mut rhs_rest = rhs.0.into_iter().map(|e| Some(e)).collect::<Vec<_>>();

        for ((_, i), (_, j)) in pairs.iter() {
            lhs_rest[*i] = None;
            rhs_rest[*j] = None;
        }

        // let axis_origin = lhs_rest
        //     .iter()
        //     .enumerate()
        //     .filter_map(|(i, e)| e.as_ref().map(|_| (0, i)))
        //     .chain(
        //         rhs_rest
        //             .iter()
        //             .enumerate()
        //             .filter_map(|(j, e)| e.as_ref().map(|_| (1, j))),
        //     )
        //     .collect();

        let merge = lhs_rest.into_iter().chain(rhs_rest).flatten().collect();

        // #[cfg(test)]
        // {
        //     std::println!("lhs_len: {}, rhs_len: {}", lhs_len, rhs_len);
        //     std::println!("axis_origin: {:?}", axis_origin);
        //     std::println!("pairs: {:?}", pairs);
        // }

        Ok((VecMapper(merge), unsafe {
            ConnectAxisOrigin::from_raw_unchecked([lhs_len, rhs_len], pairs)
        }))
    }
}

unsafe impl<'a, Id: Eq + Clone, K: Iterator<Item = &'a Id> + ExactSizeIterator>
    GroupMapper<2, LegSetArg<K>> for VecMapper<Id>
where
    Id: 'a,
{
    type Grouped = SplitBroker<Id>;
    type Err = Infallible;

    fn split(self, queue: LegSetArg<K>) -> Result<(Self::Grouped, GroupedAxes<2>), Self::Err> {
        let len = self.naxes();

        let legs = queue.into_raw();

        let mut arr = self.0.into_iter().map(|x| Some(x)).collect::<Vec<_>>();

        let mut first_ids = Vec::new();
        let mut second_ids = Vec::new();
        let mut first_idxs = Vec::new();
        let mut second_idxs = Vec::new();

        for leg in legs.into_iter() {
            if let Some(pos) = arr
                .iter()
                .position(|e| e.as_ref().map(|e| e == leg).unwrap_or(false))
            {
                let id = arr[pos].take().unwrap();
                first_ids.push(id);
                first_idxs.push(pos);
            }
        }
        for (i, e) in arr.into_iter().enumerate() {
            if let Some(id) = e {
                second_ids.push(id);
                second_idxs.push(i);
            }
        }
        Ok((
            SplitBroker {
                first: first_ids,
                second: second_ids,
            },
            unsafe { GroupedAxes::from_raw_unchecked(len, [first_idxs, second_idxs]) },
        ))
    }
}

pub struct SplitBroker<Id> {
    first: Vec<Id>,
    second: Vec<Id>,
}

unsafe impl<Id: Eq + Clone> GroupedMapper<2> for SplitBroker<Id> {
    type Mapper = VecMapper<Id>;
}

unsafe impl<const M: usize, Id: Eq + Clone> DecompGroupedMapper<2, M> for SplitBroker<Id> {
    type Err = Infallible;
    fn decomp(
        self,
        conf: DecompConf<2, M, <Self::Mapper as AxisMapper>::Id>,
    ) -> Result<[Self::Mapper; M], Self::Err> {
        let (group_belongs, new_bonds) = conf.into_raw();

        let mut x: [Vec<Id>; M] = core::array::from_fn(|_| Vec::new());
        for ((g1, id1), (g2, id2)) in new_bonds.into_iter() {
            x[g1].push(id1);
            x[g2].push(id2);
        }
        let [first, second] = group_belongs;
        x[first].extend(self.first);
        x[second].extend(self.second);

        Ok(x.map(|v| unsafe { VecMapper::from_raw_unchecked(v) }))
    }
}

impl<Id: Eq + Clone> ReplaceMapper for VecMapper<Id> {
    type Err = Id;
    fn replace(self, old_leg: &Self::Id, new_leg: Self::Id) -> Result<Self, Self::Err> {
        let mut v = self.0;
        if !v.iter().any(|e| e == &new_leg)
            && let Some(o) = v.iter_mut().find(|e| *e == old_leg)
        {
            *o = new_leg;
            return Ok(unsafe { Self::from_raw_unchecked(v) });
        }
        Err(new_leg)
    }
}

#[derive(Debug, Error)]
#[error("translation error")]
pub struct TranslateErr;

impl<Id: Eq + Clone, C> TranslateMapper<C> for VecMapper<Id> {
    type Err = TranslateErr;
    fn translate<
        'a,
        K: ExactSizeIterator + Iterator<Item = &'a Self::Id>,
        V: ExactSizeIterator + Iterator<Item = C>,
    >(
        &self,
        map: LegMapArg<K, V>,
    ) -> Result<Vec<C>, Self::Err>
    where
        Self::Id: 'a,
    {
        let (legs, content) = map.into_raw();
        if legs.len() != self.0.len() || content.len() != self.0.len() {
            return Err(TranslateErr);
        }

        let mut contents = (0..self.0.len()).map(|_| None).collect::<Vec<_>>();

        for (leg, content) in legs.zip(content) {
            match self.0.iter().position(|e| e == leg) {
                Some(idx) => {
                    if contents[idx].is_some() {
                        return Err(TranslateErr);
                    } else {
                        contents[idx] = Some(content);
                    }
                }
                None => return Err(TranslateErr),
            }
        }

        Ok(contents.into_iter().map(|e| e.unwrap()).collect())
    }
}
