#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

mod tenalg;

use core::borrow::Borrow;

use tensory_core::{
    ops::index::{ElemGetImpl, ElemGetMutImpl},
    tensor::{ConnectAxisOrigin, Tensor},
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

pub struct NdDenseRepr<E> {
    data: ArrayD<E>,
}

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
    fn view(&self) -> NdDenseViewRepr<E> {
        NdDenseViewRepr {
            data: self.data.view(),
        }
    }
    fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> NdDenseRepr<E2> {
        NdDenseRepr {
            data: self.data.map(|e| f(e)),
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

        for (idx_l, idx_r) in contr_pairs.iter() {
            if *idx_l >= lhs_raw.shape().len()
                || *idx_r >= rhs_raw.shape().len()
                || lhs_raw.shape()[*idx_l] != rhs_raw.shape()[*idx_r]
            {
                return Err(TenalgError::InvalidInput);
            }
        }

        let mut x_idxv: Vec<(bool, usize, usize)> =
            (0..lhs_raw.shape().len()).map(|i| (false, 0, i)).collect();
        let mut y_idxv: Vec<(bool, usize, usize)> =
            (0..rhs_raw.shape().len()).map(|i| (true, 0, i)).collect();

        for (num, (l, r)) in idxs_contracted.iter().enumerate() {
            x_idxv[*l].0 = true;
            y_idxv[*r].0 = false;
            x_idxv[*l].1 = num;
            y_idxv[*r].1 = num;
        }
        x_idxv.sort();
        y_idxv.sort();

        let x_idxv_ordered: Vec<usize> = x_idxv.iter().map(|(_, _, i)| *i).collect();
        let y_idxv_ordered: Vec<usize> = y_idxv.iter().map(|(_, _, i)| *i).collect();

        let mut x_idxv_to_be_ordered = (0..lhs_raw.shape().len()).collect::<Vec<_>>();
        let mut y_idxv_to_be_ordered = (0..rhs_raw.shape().len()).collect::<Vec<_>>();

        let mut x_rot = lhs_raw;
        let mut y_rot = rhs_raw;

        // println!("{:?}", x_rot.dim());
        // println!("{:?}", x_idxv_to_be_ordered);
        // println!("{:?}", y_rot.dim());
        // println!("{:?}", y_idxv_to_be_ordered);

        for i in 0..x_idxv_to_be_ordered.len() {
            for j in i..x_idxv_to_be_ordered.len() {
                if x_idxv_to_be_ordered[j] == x_idxv_ordered[i] {
                    x_idxv_to_be_ordered.swap(i, j);
                    x_rot.swap_axes(i, j);
                    //println!("{:?}", x_rot.dim());
                }
            }
        }

        for i in 0..y_idxv_to_be_ordered.len() {
            for j in i..y_idxv_to_be_ordered.len() {
                if y_idxv_to_be_ordered[j] == y_idxv_ordered[i] {
                    y_idxv_to_be_ordered.swap(i, j);
                    y_rot.swap_axes(i, j);
                    //println!("{:?}", y_rot.dim());
                }
            }
        }
        // println!("{:?}", x_rot.dim());
        // println!("{:?}", y_rot.dim());

        // println!("{:?}", x_idxv_ordered);
        // println!("{:?}", y_idxv_ordered);
        // println!("{:?}", x_idxv_to_be_ordered);
        // println!("{:?}", y_idxv_to_be_ordered);
        //println!("{}",x_rot.dim());
        //println!("{}",x_rot.dim());

        let conn_idxn = idxs_contracted.len();

        let x_rem_idxa = &x_idxv_to_be_ordered.as_slice()[0..x_idxv_ordered.len() - conn_idxn];
        let y_rem_idxa = &y_idxv_to_be_ordered.as_slice()[conn_idxn..];

        let z_idxv = x_rem_idxa
            .iter()
            .map(|idx| MulAxisOrigin::Lhs(*idx))
            .chain(y_rem_idxa.iter().map(|idx| MulAxisOrigin::Rhs(*idx)))
            .collect();
        let z = mul(&x_rot, &y_rot, conn_idxn)?;
        Ok((NdDenseRepr { data: z }, z_idxv))
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
    use std::println;

    use num_complex::Complex;
    use num_traits::abs;

    use super::*;

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

        let ta = NdDenseTensor::<_, f64>::random(leg![a=>a_n, i=>i_n, b=>b_n, j=>j_n].unwrap());
        let tb = NdDenseTensor::<_, f64>::random(leg![j=>j_n, c=>c_n, d=>d_n, i=>i_n].unwrap());

        let alv = ta.leg_alloc();
        let blv = tb.leg_alloc();
        println!("{:?}", alv);
        println!("{:?}", blv);

        //println!("{:?}", tc.leg_alloc());

        let tc = (ta.view() * tb.view()).with(())?;

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
                                tc_e += ta
                                    .get(leg_ref![&a=>ai, &i=>ii, &b=>bi, &j=>ji].unwrap())?
                                    * tb.get(leg_ref![&j=>ji, &c=>ci, &d=>di, &i=>ii].unwrap())?;
                            }
                        }
                        let tc_r = tc.get(leg_ref![&a=>ai, &b=>bi, &c=>ci, &d=>di].unwrap())?;
                        //tcd[[ai, bi, ci, di]];
                        println!("{},{},{},{} : {} vs {}", ai, bi, ci, di, tc_e, tc_r);
                        assert!(abs(tc_e - tc_r) < EPS);
                    }
                }
            }
        }

        let clv = tc.leg_alloc();
        println!("{:?}", clv);

        Ok(())
    }

    #[test]
    fn tensor_svd_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        let c = Leg::new();
        let d = Leg::new();
        let a_n = 3;
        let b_n = 4;
        let c_n = 5;
        let d_n = 6;

        let t =
            NdDenseTensor::<_, Complex<f64>>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n].unwrap());

        let us = Leg::new();
        let vs = Leg::new();

        let (u, s, v) = t.view().svd(leg_ref![&a, &b].unwrap(), us, vs)?.with(())?;

        let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        println!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", a, b, c, d, us, vs);

        println!(
            "{:?}\n{:?}\n{:?}\n",
            u.leg_alloc(),
            s.leg_alloc(),
            v.leg_alloc()
        );

        println!(
            "{:?} {:?} {:?}",
            u.raw().data.shape(),
            s.raw().data.shape(),
            v.raw().data.shape()
        );

        let us_tmp = (u.view() * s.view()).with(())?;

        println!("{:?} {:?}", us_tmp.leg_alloc(), u.raw().data.shape());

        let usv = (us_tmp.view() * v.view()).with(())?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                for ci in 0..c_n {
                    for di in 0..d_n {
                        assert!(
                            (t.get(leg_ref![&a=>ai,&b=>bi,&c=>ci,&d=>di].unwrap())?
                                - usv.get(leg_ref![&a=>ai,&b=>bi,&c=>ci,&d=>di].unwrap())?)
                            .abs()
                                < EPS
                        );
                    }
                }
            }
        }

        println!("ut");
        let mut ut = u.view().conj().with(())?;

        println!("{:?} {:?} {:?}:  {:?}", a, b, us, ut.leg_alloc());

        ut.replace_leg(&a, a.prime()).unwrap();
        ut.replace_leg(&b, b.prime()).unwrap();
        println!("{:?}", ut.leg_alloc());
        let uut = (u.view() * ut.view()).with(())?;
        println!("{:?}", uut.leg_alloc());
        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        let re = *uut.get(
                            leg_ref![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi].unwrap(),
                        )?;
                        if ai == api && bi == bpi {
                            assert!((re - 1.).abs() < EPS);
                        } else {
                            assert!(re.abs() < EPS);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
