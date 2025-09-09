//impl<Id: LegId> LegId for Prime<Id> {}

// fn vec_to_maybe_uninit_vec<T>(v: Vec<T>) -> Vec<MaybeUninit<T>> {
//     let mut v = ManuallyDrop::new(v);
//     let p = v.as_mut_ptr();
//     let len = v.len();
//     let cap = v.capacity();
//     unsafe { Vec::from_raw_parts(p as _, len, cap) }
// }

use core::convert::Infallible;
use thiserror::Error;

use alloc::vec::Vec;

use tensory_core::tensor::{
    BuildableBroker, ConnectAxisOrigin, ConnectBroker, DecompConf, DecompGroupedBroker,
    GroupBroker, GroupedAxes, GroupedBroker, LegMapArg, LegSetArg, ReplaceBroker, TensorBroker,
    TranslateBroker,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecBroker<T>(Vec<T>);
impl<T> VecBroker<T> {
    pub unsafe fn from_raw_unchecked(raw: Vec<T>) -> Self {
        VecBroker(raw)
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

impl<T: Eq + Clone> TensorBroker for VecBroker<T> {
    type Id = T;

    fn len(&self) -> usize {
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

impl<T: Eq + Clone, I: Iterator<Item = Self::Id>> BuildableBroker<I> for VecBroker<T> {
    type Err = BuildErr;
    fn build(iter: I) -> Result<Self, Self::Err> {
        let v = iter.collect();
        Self::from_raw(v).map_err(|_| BuildErr)
    }
}

// unsafe impl<T: Eq + Clone> OverlayBroker<2> for VecBroker<T> {
//     type Err;

//     fn overlay(mgrs: [Self; 2]) -> Result<(Self, Vec<[usize; 2]>), Self::Err> {
//         todo!()
//     }
// }

unsafe impl<T: Eq + Clone> ConnectBroker<2> for VecBroker<T> {
    type Err = Infallible;
    fn connect([lhs, rhs]: [Self; 2]) -> Result<(Self, ConnectAxisOrigin<2>), Self::Err> {
        let lhs_len = lhs.len();
        let rhs_len = rhs.len();

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

        let merge = lhs_rest
            .into_iter()
            .chain(rhs_rest.into_iter())
            .flatten()
            .collect();

        // #[cfg(test)]
        // {
        //     std::println!("lhs_len: {}, rhs_len: {}", lhs_len, rhs_len);
        //     std::println!("axis_origin: {:?}", axis_origin);
        //     std::println!("pairs: {:?}", pairs);
        // }

        Ok((VecBroker(merge), unsafe {
            ConnectAxisOrigin::from_raw_unchecked([lhs_len, rhs_len], pairs)
        }))
    }
}

unsafe impl<'a, Id: Eq + Clone, K: Iterator<Item = &'a Id> + ExactSizeIterator>
    GroupBroker<2, LegSetArg<K>> for VecBroker<Id>
where
    Id: 'a,
{
    type Grouped = SplitBroker<Id>;
    type Err = Infallible;

    fn split(
        self,
        queue: LegSetArg<K>,
    ) -> Result<(Self::Grouped, tensory_core::tensor::GroupedAxes<2>), Self::Err> {
        let len = self.len();

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
                let id = core::mem::replace(&mut arr[pos], None).unwrap();
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

unsafe impl<Id: Eq + Clone> GroupedBroker<2> for SplitBroker<Id> {
    type Broker = VecBroker<Id>;
}

unsafe impl<const M: usize, Id: Eq + Clone> DecompGroupedBroker<2, M> for SplitBroker<Id> {
    type Err = Infallible;
    fn decomp(
        self,
        conf: DecompConf<2, M, <Self::Broker as TensorBroker>::Id>,
    ) -> Result<[Self::Broker; M], Self::Err> {
        let (group_belongs, new_bonds) = conf.into_raw();

        let mut x: [Vec<Id>; M] = core::array::from_fn(|_| Vec::new());
        for ((g1, id1), (g2, id2)) in new_bonds.into_iter() {
            x[g1].push(id1);
            x[g2].push(id2);
        }
        let [first, second] = group_belongs;
        x[first].extend(self.first);
        x[second].extend(self.second);

        Ok(x.map(|v| unsafe { VecBroker::from_raw_unchecked(v) }))
    }
}

impl<Id: Eq + Clone> ReplaceBroker for VecBroker<Id> {
    type Err = Id;
    fn replace(self, old_leg: &Self::Id, new_leg: Self::Id) -> Result<Self, Self::Err> {
        let mut v = self.0;
        if let None = v.iter().position(|e| e == &new_leg) {
            if let Some(o) = v.iter_mut().find(|e| *e == old_leg) {
                *o = new_leg;
                return Ok(unsafe { Self::from_raw_unchecked(v) });
            }
        }
        Err(new_leg)
    }
}

// unsafe impl<Set, T: Eq> SvdLegAlloc<Set> for VecLegAlloc<T> {
//     type Intermediate = (Vec<MaybeUninit<T>>, T, T, T, T);

//     fn extract(
//         a: Self,
//         u_legs: Set,
//         u_us_leg: Self::Id,
//         s_us_leg: Self::Id,
//         s_sv_leg: Self::Id,
//         v_sv_leg: Self::Id,
//     ) -> (Self::Intermediate, Vec<usize>) {
//         let v = Vec::new();

//         ((
//             vec_to_maybe_uninit_vec(a.0),
//             u_us_leg,
//             s_us_leg,
//             s_sv_leg,
//             v_sv_leg,
//         ))
//     }

//     unsafe fn merge(
//         intermediate: Self::Intermediate,
//         u_provenance: Vec<crate::core::svd::SvdIsometryAxisProvenance>,
//         s_provenance: crate::core::svd::SvdSingularAxisOrder,
//         v_provenance: Vec<crate::core::svd::SvdIsometryAxisProvenance>,
//     ) -> (Self, Self, Self)
//     where
//         Self: Sized,
//     {
//         let (a_legs, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg) = intermediate;
//         let u_us_leg = MaybeUninit::new(u_us_leg);
//         let v_sv_leg = MaybeUninit::new(v_sv_leg);
//         let u_legs = u_provenance
//             .into_iter()
//             .map(|prov| match prov {
//                 SvdIsometryAxisProvenance::Original(idx) => unsafe {
//                     a_legs[idx].assume_init_read()
//                 },
//                 SvdIsometryAxisProvenance::Singular => unsafe { u_us_leg.assume_init_read() },
//             })
//             .collect();

//         let v_legs = v_provenance
//             .into_iter()
//             .map(|prov| match prov {
//                 SvdIsometryAxisProvenance::Original(idx) => unsafe {
//                     a_legs[idx].assume_init_read()
//                 },
//                 SvdIsometryAxisProvenance::Singular => unsafe { v_sv_leg.assume_init_read() },
//             })
//             .collect();

//         let s_legs = match s_provenance {
//             SvdSingularAxisOrder::UV => [s_us_leg, s_sv_leg].into_iter().collect(),
//             SvdSingularAxisOrder::VU => [s_sv_leg, s_us_leg].into_iter().collect(),
//         };

//         (Self(u_legs), Self(s_legs), Self(v_legs))
//     }
// }

// unsafe impl<T> ConjugationLegAlloc for VecLegAlloc<T> {
//     unsafe fn merge(
//         self,
//         axis_provenance: Vec<crate::core::conjugation::ConjugationAxisProvenance>,
//     ) -> Self {
//         let legs = vec_to_maybe_uninit_vec(self.0);

//         let aconj_legs = axis_provenance
//             .into_iter()
//             .map(|ConjugationAxisProvenance(idx)| unsafe { legs[idx].assume_init_read() })
//             .collect();

//         Self(aconj_legs)
//     }
// }

#[derive(Debug, Error)]
#[error("translation error")]
pub struct TranslateErr;

impl<Id: Eq + Clone, C> TranslateBroker<C> for VecBroker<Id> {
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
