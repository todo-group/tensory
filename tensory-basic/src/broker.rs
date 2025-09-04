//impl<Id: LegId> LegId for Prime<Id> {}

// fn vec_to_maybe_uninit_vec<T>(v: Vec<T>) -> Vec<MaybeUninit<T>> {
//     let mut v = ManuallyDrop::new(v);
//     let p = v.as_mut_ptr();
//     let len = v.len();
//     let cap = v.capacity();
//     unsafe { Vec::from_raw_parts(p as _, len, cap) }
// }

use core::{
    convert::Infallible,
    mem::{ManuallyDrop, MaybeUninit},
};

use alloc::vec::Vec;

use tensory_core::tensor::{ConnectAxisOrigin, ConnectBroker, OverlayBroker, TensorBroker};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VecBroker<T>(Vec<T>);
impl<T> VecBroker<T> {
    pub fn from_raw(raw: Vec<T>) -> Self {
        VecBroker(raw)
    }
}

impl<T: Eq + Clone> TensorBroker for VecBroker<T> {
    type Id = T;

    fn len(&self) -> usize {
        self.0.len()
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
    fn connect(
        [lhs, rhs]: [Self; 2],
    ) -> Result<(Self, tensory_core::tensor::ConnectAxisOrigin<2>), Self::Err> {
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

        let axis_origin = lhs_rest
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|_| (0, i)))
            .chain(
                rhs_rest
                    .iter()
                    .enumerate()
                    .filter_map(|(j, e)| e.as_ref().map(|_| (1, j))),
            )
            .collect();

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
            ConnectAxisOrigin::from_raw_unchecked([lhs_len, rhs_len], axis_origin, pairs)
        }))
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
