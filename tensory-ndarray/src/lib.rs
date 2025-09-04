#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

mod tenalg;

use core::{borrow::Borrow, panic};

use tensory_core::{
    ops::index::{ElemGetImpl, ElemGetMutImpl},
    tensor::{ConnectAxisOrigin, Tensor, ViewableRepr},
};

use alloc::vec::Vec;
use ndarray::{ArrayBase, ArrayD, ArrayViewD, IxDyn, OwnedRepr};
use ndarray_linalg::{Lapack, Scalar, from_diag, random};
use num_traits::{ConstZero, Zero};
use tensory_core::{
    ops::contr::{MulAxisOrigin, MulCtxImpl},
    tensor::TensorRepr,
};

use crate::tenalg::{conj, cut_filter::CutFilter, error::TenalgError, into_svd, mul};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseRepr<E> {
    data: ArrayD<E>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseViewRepr<'a, E> {
    data: ArrayViewD<'a, E>,
}

impl<E> NdDenseRepr<E> {
    fn random(sizes: impl IntoIterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.into_iter().collect();
        Self {
            data: random(sizes),
        }
    }
    fn zero(sizes: impl IntoIterator<Item = usize>) -> Self
    where
        E: Clone + Zero,
    {
        let sizes: Vec<usize> = sizes.into_iter().collect();
        Self {
            data: ArrayD::zeros(sizes),
        }
    }
    fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> NdDenseRepr<E2> {
        NdDenseRepr {
            data: self.data.map(|e| f(e)),
        }
    }
}

impl<'a, E: 'a> ViewableRepr<'a> for NdDenseRepr<E> {
    type View = NdDenseViewRepr<'a, E>;
    fn view(&'a self) -> Self::View {
        NdDenseViewRepr {
            data: self.data.view(),
        }
    }
}

unsafe impl<E> TensorRepr for NdDenseRepr<E> {
    fn dim(&self) -> usize {
        self.data.shape().len()
    }
}
unsafe impl<E> TensorRepr for NdDenseViewRepr<'_, E> {
    fn dim(&self) -> usize {
        self.data.shape().len()
    }
}
// impl<E> AxisInfoImpl for NdDenseRepr<E> {
//     type AxisInfo = usize;

//     unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
//         self.data.shape()[i]
//     }
// }

// impl<E> AxisInfoImpl for NdDenseViewRepr<'_, E> {
//     type AxisInfo = usize;

//     unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
//         self.data.shape()[i]
//     }
// }

// unsafe impl<E: Scalar> ConjCtx<NdDenseViewRepr<'_, E>> for () {
//     type Res = NdDenseRepr<E>;

//     type Err = TenalgError;

//     fn conjugate(
//         self,
//         a: NdDenseViewRepr<'_, E>,
//     ) -> core::result::Result<(NdDenseRepr<E>, Vec<ConjugationAxisProvenance>), TenalgError> {
//         Ok((
//             NdDenseRepr {
//                 data: conj(&a.data)?,
//             },
//             (0..a.data.shape().len())
//                 .map(|i| ConjugationAxisProvenance(i))
//                 .collect(),
//         ))
//     }
// }

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdDenseReprError;
impl core::error::Error for NdDenseReprError {}
impl core::fmt::Display for NdDenseReprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DenseRepr operation failed")
    }
}

//struct NdDenseContext;

unsafe impl<'a, E: Lapack + Scalar + ConstZero>
    MulCtxImpl<NdDenseViewRepr<'a, E>, NdDenseViewRepr<'a, E>> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    unsafe fn mul_unchecked(
        self,
        lhs: NdDenseViewRepr<'a, E>,
        rhs: NdDenseViewRepr<'a, E>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        let ([lhs_dim, rhs_dim], res_origin, contr_pairs) = axis_origin.into_raw();

        let mut lhs_idxv: Vec<(bool, usize, usize)> = (0..lhs_dim).map(|i| (false, 0, i)).collect(); // true is connect
        let mut rhs_idxv: Vec<(bool, usize, usize)> = (0..rhs_dim).map(|i| (true, 0, i)).collect(); // false is connect

        for (k, ((t1, idx1), (t2, idx2))) in contr_pairs.iter().enumerate() {
            let (lhs_idx, rhs_idx) = match (t1, t2) {
                (0, 1) => (idx1, idx2),
                (1, 0) => (idx2, idx1),
                _ => panic!("Unexpected axis"),
            };
            if lhs_raw.shape()[*lhs_idx] != rhs_raw.shape()[*rhs_idx] {
                return Err(TenalgError::InvalidInput);
            }
            lhs_idxv[*lhs_idx].0 = true;
            rhs_idxv[*rhs_idx].0 = false;
            lhs_idxv[*lhs_idx].1 = k;
            rhs_idxv[*rhs_idx].1 = k;
        }
        lhs_idxv.sort();
        rhs_idxv.sort();

        let lhs_idxv_ordered: Vec<usize> = lhs_idxv.iter().map(|(_, _, i)| *i).collect();
        let rhs_idxv_ordered: Vec<usize> = rhs_idxv.iter().map(|(_, _, i)| *i).collect();

        let lhs_rot = lhs_raw.permuted_axes(lhs_idxv_ordered.as_slice());
        let rhs_rot = rhs_raw.permuted_axes(rhs_idxv_ordered.as_slice());

        let conn_idxn = contr_pairs.len();
        let z = mul(&lhs_rot, &rhs_rot, conn_idxn)?;

        let lhs_rem_idxa = &lhs_idxv_ordered.as_slice()[0..lhs_idxv_ordered.len() - conn_idxn];
        let rhs_rem_idxa = &rhs_idxv_ordered.as_slice()[conn_idxn..];

        #[cfg(test)]
        {
            use std::println;
            println!("{:?}", lhs_rot.dim());
            println!("{:?}", lhs_idxv_ordered);
            println!("{:?}", rhs_rot.dim());
            println!("{:?}", rhs_idxv_ordered);

            println!("{:?}", lhs_rem_idxa);
            println!("{:?}", rhs_rem_idxa);
        }

        let mut z_idxv: Vec<(usize, usize)> = (0..z.shape().len()).map(|i| (0, i)).collect();

        res_origin.into_iter().enumerate().for_each(|(k, (t, i))| {
            z_idxv[match t {
                0 => lhs_rem_idxa.iter().position(|&j| i == j).unwrap(),
                1 => rhs_rem_idxa.iter().position(|&j| i == j).unwrap() + lhs_rem_idxa.len(),
                _ => panic!("Unexpected axis"),
            }]
            .0 = k;
        });

        z_idxv.sort();

        let z_idxv = z_idxv.into_iter().map(|(_, i)| i).collect::<Vec<_>>();

        let z = z.permuted_axes(z_idxv);

        // let z_idxv = x_rem_idxa
        //     .iter()
        //     .map(|idx| MulAxisOrigin::Lhs(*idx))
        //     .chain(y_rem_idxa.iter().map(|idx| MulAxisOrigin::Rhs(*idx)))
        //     .collect();

        Ok(NdDenseRepr { data: z })
    }
}

// unsafe impl<'a, E: Scalar + Lapack, C: CutFilter<<E as Scalar>::Real>>
//     SvdContextImpl<NdDenseViewRepr<'a, E>> for C
// where
//     E::Real: ConstZero,
// {
//     type U = NdDenseRepr<E>;
//     type S = NdDenseRepr<E::Real>;
//     type V = NdDenseRepr<E>;
//     type Err = TenalgError;

//     unsafe fn svd_unchecked(
//         self,
//         a: NdDenseViewRepr<'a, E>,
//         u_legs: &[usize],
//     ) -> Result<
//         (
//             Self::U,
//             Self::S,
//             Self::V,
//             Vec<SvdIsometryAxisProvenance>,
//             SvdSingularAxisOrder,
//             Vec<SvdIsometryAxisProvenance>,
//         ),
//         Self::Err,
//     > {
//         let a_raw = a.data;

//         let mut a_idxv: Vec<(bool, usize)> = (0..a_raw.shape().len()).map(|i| (true, i)).collect();

//         for u_leg in u_legs.iter() {
//             a_idxv[*u_leg].0 = false;
//         }

//         a_idxv.sort();

//         let a_idxv_ordered: Vec<usize> = a_idxv.iter().map(|(_, i)| *i).collect();

//         let mut a_idxv_to_be_ordered = (0..a_raw.shape().len()).collect::<Vec<_>>();

//         let mut a_rot = a_raw;

//         for i in 0..a_idxv_to_be_ordered.len() {
//             for j in i..a_idxv_to_be_ordered.len() {
//                 if a_idxv_to_be_ordered[j] == a_idxv_ordered[i] {
//                     a_idxv_to_be_ordered.swap(i, j);
//                     a_rot.swap_axes(i, j);
//                     //println!("{:?}", self_rot.dim());
//                 }
//             }
//         }
//         let (u, s, v) = into_svd(a_rot, u_legs.len(), self)?;

//         let s: ArrayBase<OwnedRepr<<E as Scalar>::Real>, ndarray::Dim<[usize; 1]>> = s;

//         let u_idxn = u_legs.len();

//         let u_rem_idxa = &a_idxv_ordered.as_slice()[0..u_idxn];
//         let v_rem_idxa = &a_idxv_ordered.as_slice()[u_idxn..];

//         let u_idxv: Vec<_> = u_rem_idxa
//             .iter()
//             .map(|idx| SvdIsometryAxisProvenance::Original(*idx))
//             .chain(core::iter::once(SvdIsometryAxisProvenance::Singular))
//             .collect();
//         let v_idxv: Vec<_> = core::iter::once(SvdIsometryAxisProvenance::Singular)
//             .chain(
//                 v_rem_idxa
//                     .iter()
//                     .map(|idx| SvdIsometryAxisProvenance::Original(*idx)),
//             )
//             .collect();

//         let s_ten = from_diag(&s.to_vec()).into_dimensionality::<IxDyn>()?;

//         Ok((
//             NdDenseRepr { data: u },
//             NdDenseRepr { data: s_ten },
//             NdDenseRepr { data: v },
//             u_idxv,
//             SvdSingularAxisOrder::UV,
//             v_idxv,
//         ))
//     }
// }

impl<E> ElemGetImpl for NdDenseRepr<E> {
    type Index = usize;
    type E = E;
    type Err = NdDenseReprError;

    unsafe fn get_unchecked(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

impl<E> ElemGetMutImpl for NdDenseRepr<E> {
    unsafe fn get_mut_unchecked(
        &mut self,
        indices: Vec<Self::Index>,
    ) -> Result<&mut Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get_mut(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

pub type NdDenseTensor<LA, E> = Tensor<LA, NdDenseRepr<E>>;

// impl<LA: LegAlloc, E> NdDenseTensor<LA, E> {
//     pub fn random(leg_size_set: LegMap<LA::Id, usize>) -> Self
//     where
//         E: Scalar,
//     {
//         let (id, size) = leg_size_set.into_raw();

//         unsafe {
//             Self::from_raw_unchecked(
//                 NdDenseRepr::random(size),
//                 LegAlloc::from_raw(id).map_err(|_| "NEVER!").unwrap(),
//             )
//         }
//     }
//     pub fn zero(leg_size_set: LegMap<LA::Id, usize>) -> Self
//     where
//         E: Clone + Zero,
//     {
//         let (id, size): (Vec<_>, Vec<_>) = leg_size_set.into_raw();
//         Self::from_raw(
//             NdDenseRepr::zero(size),
//             LegAlloc::from_raw(id).map_err(|_| "NEVER!").unwrap(),
//         )
//         .map_err(|_| "why!?")
//         .unwrap()
//     }
//     pub fn view(&self) -> Tensor<LA, NdDenseViewRepr<E>>
//     where
//         LA: Clone,
//     {
//         Tensor::from_raw(self.raw().view(), self.leg_alloc().clone())
//             .map_err(|_| "why!?")
//             .unwrap()
//     }
//     pub fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> Tensor<LA, NdDenseRepr<E2>>
//     where
//         LA: Clone,
//     {
//         Tensor::from_raw(self.raw().map(&mut f), self.leg_alloc().clone())
//             .map_err(|_| "why!?")
//             .unwrap()
//     }
// }

#[cfg(test)]
mod tests {
    use std::{println, vec};

    use num_complex::Complex;
    use num_traits::abs;
    use tensory_core::ops::index::ElemGet;

    use super::*;

    use tensory_basic::{
        broker::VecBroker,
        id::{Id128, Prime},
    };

    const EPS: f64 = 1e-8;

    type Leg = Prime<Id128>;

    #[test]
    fn tensor_mul_test() -> Result<(), anyhow::Error> {
        let a_n = 3;
        let b_n = 4;
        let c_n = 5;
        let d_n = 6;
        let i_n = 7;
        let j_n = 8;

        let a = Leg::new();
        let b = Leg::new();
        let c = Leg::new();
        let d = Leg::new();
        let i = Leg::new();
        let j = Leg::new();

        let ta = unsafe {
            Tensor::from_raw_unchecked(
                NdDenseRepr::<f64>::random([a_n, i_n, b_n, j_n]),
                VecBroker::from_raw(vec![a, i, b, j]),
            )
        };
        let tb = unsafe {
            Tensor::from_raw_unchecked(
                NdDenseRepr::random([j_n, c_n, d_n, i_n]),
                VecBroker::from_raw(vec![j, c, d, i]),
            )
        };

        let alv = ta.broker();
        let blv = tb.broker();
        println!("{:?}", alv);
        println!("{:?}", blv);

        //println!("{:?}", tc.leg_alloc());

        let tc = (&ta * &tb)?;
        println!("tc: {:?}", tc);
        let tc = tc.with(())?;

        let clv = tc.broker();
        println!("{:?}", clv);

        let (ta, _) = ta.into_raw();
        let (tb, _) = tb.into_raw();
        let (tc, _) = tc.into_raw();

        //let tad = ta.data();
        //let tbd = tb.data();
        //let tcd = tc.data();
        //println!("{:?} {:?} {:?}", tad.shape(), tbd.shape(), tcd.shape());
        for ai in 0..3 {
            for bi in 0..4 {
                for ci in 0..5 {
                    for di in 0..6 {
                        let mut tc_e = 0.0;
                        for ii in 0..7 {
                            for ji in 0..8 {
                                tc_e +=
                                    ta.get(vec![ai, ii, bi, ji])? * tb.get(vec![ji, ci, di, ii])?;
                            }
                        }
                        let tc_r = tc.get(vec![ai, bi, ci, di])?;
                        //tcd[[ai, bi, ci, di]];
                        println!("{},{},{},{} : {} vs {}", ai, bi, ci, di, tc_e, tc_r);
                        assert!(abs(tc_e - tc_r) < EPS);
                    }
                }
            }
        }

        Ok(())
    }

    // #[test]
    // fn tensor_svd_test() -> anyhow::Result<()> {
    //     let a = Leg::new();
    //     let b = Leg::new();
    //     let c = Leg::new();
    //     let d = Leg::new();
    //     let a_n = 3;
    //     let b_n = 4;
    //     let c_n = 5;
    //     let d_n = 6;

    //     let t =
    //         NdDenseTensor::<_, Complex<f64>>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n].unwrap());

    //     let us = Leg::new();
    //     let vs = Leg::new();

    //     let (u, s, v) = t.view().svd(leg_ref![&a, &b].unwrap(), us, vs)?.with(())?;

    //     let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

    //     println!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", a, b, c, d, us, vs);

    //     println!(
    //         "{:?}\n{:?}\n{:?}\n",
    //         u.leg_alloc(),
    //         s.leg_alloc(),
    //         v.leg_alloc()
    //     );

    //     println!(
    //         "{:?} {:?} {:?}",
    //         u.raw().data.shape(),
    //         s.raw().data.shape(),
    //         v.raw().data.shape()
    //     );

    //     let us_tmp = (u.view() * s.view()).with(())?;

    //     println!("{:?} {:?}", us_tmp.leg_alloc(), u.raw().data.shape());

    //     let usv = (us_tmp.view() * v.view()).with(())?;

    //     for ai in 0..a_n {
    //         for bi in 0..b_n {
    //             for ci in 0..c_n {
    //                 for di in 0..d_n {
    //                     assert!(
    //                         (t.get(leg_ref![&a=>ai,&b=>bi,&c=>ci,&d=>di].unwrap())?
    //                             - usv.get(leg_ref![&a=>ai,&b=>bi,&c=>ci,&d=>di].unwrap())?)
    //                         .abs()
    //                             < EPS
    //                     );
    //                 }
    //             }
    //         }
    //     }

    //     println!("ut");
    //     let mut ut = u.view().conj().with(())?;

    //     println!("{:?} {:?} {:?}:  {:?}", a, b, us, ut.leg_alloc());

    //     ut.replace_leg(&a, a.prime()).unwrap();
    //     ut.replace_leg(&b, b.prime()).unwrap();
    //     println!("{:?}", ut.leg_alloc());
    //     let uut = (u.view() * ut.view()).with(())?;
    //     println!("{:?}", uut.leg_alloc());
    //     for ai in 0..a_n {
    //         for bi in 0..b_n {
    //             for api in 0..a_n {
    //                 for bpi in 0..b_n {
    //                     let re = *uut.get(
    //                         leg_ref![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi].unwrap(),
    //                     )?;
    //                     if ai == api && bi == bpi {
    //                         assert!((re - 1.).abs() < EPS);
    //                     } else {
    //                         assert!(re.abs() < EPS);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     Ok(())
    // }
}
