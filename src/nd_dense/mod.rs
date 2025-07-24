mod tenalg;

use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    ops::MulAssign,
};

use ndarray::{
    ArrayBase, ArrayD, ArrayView, ArrayViewD, Dimension, IxDyn, LinalgScalar, OwnedRepr,
    ScalarOperand,
};
use ndarray_linalg::{Lapack, Scalar, from_diag, random};
use num_traits::{ConstZero, Zero};

use crate::{
    core::{
        tensor::{LegId, LegVal, Tensor},
        tensor_repr::{ContractionContext, ContractionIndexProvenance, ElementAccess, TensorRepr},
    },
    nd_dense::tenalg::{error::TenalgError, mul},
};

// use crate::{
//     Tensor,
//     leg::{LegId, LegVal},
//     raw_tensor::{Contract, ElementAccess, RawTensor},
//     tenalg::{conj, cut_filter::CutFilter, into_svd, mul, svd},
// };

//type Result<T> = std::result::Result<T, TensorError>;

pub struct NdDenseRepr<E> {
    data: ArrayD<E>,
}

pub struct NdDenseViewRepr<'a, E> {
    data: ArrayViewD<'a, E>,
}

impl<E> NdDenseRepr<E> {
    fn random(indices: impl IntoIterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let index_set: Vec<usize> = indices.into_iter().collect();
        Self {
            data: random(index_set),
        }
    }
    fn zero(indices: impl IntoIterator<Item = usize>) -> Self
    where
        E: Clone + Zero,
    {
        let index_set: Vec<usize> = indices.into_iter().collect();
        Self {
            data: ArrayD::zeros(index_set),
        }
    }
    fn view(&self) -> NdDenseViewRepr<E> {
        NdDenseViewRepr {
            data: self.data.view(),
        }
    }
}

impl<E> TensorRepr for NdDenseRepr<E> {
    fn len(&self) -> usize {
        self.data.shape().len()
    }
}
impl<'a, E> TensorRepr for NdDenseViewRepr<'a, E> {
    fn len(&self) -> usize {
        self.data.shape().len()
    }
}

// impl<E> Indexable for DenseRepr<E> {
//     type Index = usize;

//     fn get(&self, locs: &[LegVal<Self::Id, Self::Index>]) -> Result<&E> {
//         let map: HashMap<_, _> = locs.iter().map(|iv| (&iv.0, iv.1)).collect();
//         let opv: Option<Vec<usize>> = self
//             .index_alloc
//             .iter()
//             .map(|idx| map.get(&idx.id).map(|loc| *loc))
//             .collect();
//         let v = opv.ok_or(TensorError::InvalidLoc)?;

//         self.data.get(v.as_slice()).ok_or(TensorError::InvalidLoc)
//     }
//     fn get_mut(&mut self, locs: &[LegVal<Self::Id, Self::Index>]) -> Result<&mut E> {
//         let map: HashMap<_, _> = locs.iter().map(|iv| (&iv.0, iv.1)).collect();
//         let opv: Option<Vec<usize>> = self
//             .index_alloc
//             .iter()
//             .map(|idx| map.get(&idx.id).map(|loc| *loc))
//             .collect();
//         let v = opv.ok_or(TensorError::InvalidLoc)?;

//         self.data
//             .get_mut(v.as_slice())
//             .ok_or(TensorError::InvalidLoc)
//     }
// }

// impl<E: Scalar, Id: LegId> TConj for DenseRepr<E, Id> {
//     type TO = Self;

//     fn d_conj(self) -> Result<Self::TO> {
//         Ok(Self {
//             index_alloc: self.index_alloc,
//             data: conj(&self.data)?,
//         })
//     }
//     fn t_conj(&self) -> Result<Self::TO> {
//         Ok(Self {
//             index_alloc: self.index_alloc.clone(),
//             data: conj(&self.data)?,
//         })
//     }
// }

// impl<E: LinalgScalar + ScalarOperand + MulAssign, Id: LegId> STMul<E> for DenseRepr<E, Id>
// where
//     E: Clone + Zero,
// {
//     fn st_mul(&mut self, scalar: E) {
//         if scalar.is_zero() {
//             self.data.fill(E::zero());
//         } else {
//             self.data *= scalar;
//         }
//     }
// }

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdDenseReprError;
impl std::error::Error for NdDenseReprError {}
impl std::fmt::Display for NdDenseReprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DenseRepr operation failed")
    }
}

//struct NdDenseContext;

impl<'a, E: Lapack + Scalar + ConstZero>
    ContractionContext<NdDenseViewRepr<'a, E>, NdDenseViewRepr<'a, E>> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    fn contract(
        self,
        lhs: NdDenseViewRepr<'a, E>,
        rhs: NdDenseViewRepr<'a, E>,
        idxs_contracted: &[(usize, usize)],
    ) -> Result<(Self::Res, Vec<(ContractionIndexProvenance, usize)>), Self::Err> {
        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        for (idx_l, idx_r) in idxs_contracted.iter() {
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
            (0..lhs_raw.shape().len()).map(|i| (true, 0, i)).collect();

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
        // println!("{:?}", x_idxv_to_be_ordered.iter().map(|idx| idx.dim()));
        // println!("{:?}", y_rot.dim());
        // println!("{:?}", y_idxv_to_be_ordered.iter().map(|idx| idx.dim()));

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
            .map(|idx| (ContractionIndexProvenance::Lhs, *idx))
            .chain(
                y_rem_idxa
                    .iter()
                    .map(|idx| (ContractionIndexProvenance::Rhs, *idx)),
            )
            .collect();
        let z = mul(&x_rot, &y_rot, conn_idxn)?;
        Ok((NdDenseRepr { data: z }, z_idxv))
    }
}

// impl<E: Lapack + Scalar + ConstZero, Id: LegIdRepr> TSVD for DenseRepr<E, Id>
// where
//     E::Real: ConstZero,
// {
//     type TU = DenseRepr<E, Id>;
//     type TS = DenseRepr<E::Real, Id>;
//     type TV = DenseRepr<E, Id>;
//     type ES = E::Real;

//     fn d_svd<F: CutFilter<Self::ES>>(
//         self,
//         u_idxa: &[Self::LegId],
//         us_idx: Self::LegId,
//         sv_idx: Self::LegId,
//         s_filter: F,
//     ) -> Result<(Self::TU, Self::TS, Self::TV)> {
//         let u_idxs: HashSet<Index> = u_idxa.iter().cloned().collect();
//         let u_idxn = u_idxa.len();

//         let mut self_idxv_marked: Vec<(bool, Index)> = self
//             .index_alloc
//             .iter()
//             .cloned()
//             .map(|idx| (!u_idxs.contains(&idx), idx))
//             .collect();
//         self_idxv_marked.sort();
//         let self_idxv_ordered: Vec<Index> =
//             self_idxv_marked.into_iter().map(|(_, idx)| idx).collect();

//         let mut self_rot = self.data.view();
//         let mut self_idxv_to_be_ordered = self.index_alloc.clone();
//         // let mut y_idxv_to_be_ordered = rhs.index_alloc.clone();

//         // // println!("{:?}", x_rot.dim());
//         // // println!("{:?}", x_idxv_to_be_ordered.iter().map(|idx| idx.dim()));
//         // // println!("{:?}", y_rot.dim());
//         // // println!("{:?}", y_idxv_to_be_ordered.iter().map(|idx| idx.dim()));

//         for i in 0..self_idxv_to_be_ordered.len() {
//             for j in i..self_idxv_to_be_ordered.len() {
//                 if self_idxv_to_be_ordered[j] == self_idxv_ordered[i] {
//                     self_idxv_to_be_ordered.swap(i, j);
//                     self_rot.swap_axes(i, j);
//                     //println!("{:?}", self_rot.dim());
//                 }
//             }
//         }
//         let (u, s, v) = into_svd(self_rot, u_idxa.len(), s_filter)?;

//         let s: ArrayBase<OwnedRepr<<E as Scalar>::Real>, ndarray::Dim<[usize; 1]>> = s;

//         let u_rem_idxa = &self_idxv_ordered.as_slice()[0..u_idxn];
//         let v_rem_idxa = &self_idxv_ordered.as_slice()[u_idxn..];

//         let s_dim = s.shape()[0];
//         let idx_us = Index::new(s_dim);
//         let idx_sv = Index::new(s_dim);
//         let u_idxv: Vec<Index> = u_rem_idxa
//             .iter()
//             .chain([idx_us.clone()].iter())
//             .cloned()
//             .collect();
//         let v_idxv: Vec<Index> = [idx_sv.clone()]
//             .iter()
//             .chain(v_rem_idxa.iter())
//             .cloned()
//             .collect();

//         let s_ten = from_diag(&s.to_vec()).into_dimensionality::<IxDyn>()?;

//         Ok((
//             Self::TU {
//                 index_alloc: u_idxv,
//                 data: u,
//             },
//             Self::TS {
//                 index_alloc: vec![idx_us, idx_sv],
//                 data: s_ten,
//             },
//             Self::TV {
//                 index_alloc: v_idxv,
//                 data: v,
//             },
//         ))
//     }

//     fn t_svd<F: CutFilter<Self::ES>>(
//         &self,
//         u_idxa: &[Self::LegId],
//         us_idx: Self::LegId,
//         sv_idx: Self::LegId,
//         s_filter: F,
//     ) -> Result<(Self::TU, Self::TS, Self::TV)> {
//         let u_idxs: HashSet<Index> = u_idxa.iter().cloned().collect();
//         let u_idxn = u_idxa.len();

//         let mut self_idxv_marked: Vec<(bool, Index)> = self
//             .index_alloc
//             .iter()
//             .cloned()
//             .map(|idx| (!u_idxs.contains(&idx), idx))
//             .collect();
//         self_idxv_marked.sort();
//         let self_idxv_ordered: Vec<Index> =
//             self_idxv_marked.into_iter().map(|(_, idx)| idx).collect();

//         let mut self_rot = self.data.view();
//         let mut self_idxv_to_be_ordered = self.index_alloc.clone();
//         // let mut y_idxv_to_be_ordered = rhs.index_alloc.clone();

//         // // println!("{:?}", x_rot.dim());
//         // // println!("{:?}", x_idxv_to_be_ordered.iter().map(|idx| idx.dim()));
//         // // println!("{:?}", y_rot.dim());
//         // // println!("{:?}", y_idxv_to_be_ordered.iter().map(|idx| idx.dim()));

//         for i in 0..self_idxv_to_be_ordered.len() {
//             for j in i..self_idxv_to_be_ordered.len() {
//                 if self_idxv_to_be_ordered[j] == self_idxv_ordered[i] {
//                     self_idxv_to_be_ordered.swap(i, j);
//                     self_rot.swap_axes(i, j);
//                     //println!("{:?}", self_rot.dim());
//                 }
//             }
//         }
//         let (u, s, v) = svd(&self_rot, u_idxa.len(), s_filter)?;

//         let s: ArrayBase<OwnedRepr<<E as Scalar>::Real>, ndarray::Dim<[usize; 1]>> = s;

//         let u_rem_idxa = &self_idxv_ordered.as_slice()[0..u_idxn];
//         let v_rem_idxa = &self_idxv_ordered.as_slice()[u_idxn..];

//         let s_dim = s.shape()[0];
//         let idx_us = Index::new(s_dim);
//         let idx_sv = Index::new(s_dim);
//         let u_idxv: Vec<Index> = u_rem_idxa
//             .iter()
//             .chain([idx_us.clone()].iter())
//             .cloned()
//             .collect();
//         let v_idxv: Vec<Index> = [idx_sv.clone()]
//             .iter()
//             .chain(v_rem_idxa.iter())
//             .cloned()
//             .collect();

//         let s_ten = from_diag(&s.to_vec()).into_dimensionality::<IxDyn>()?;

//         Ok((
//             Self::TU {
//                 index_alloc: u_idxv,
//                 data: u,
//             },
//             Self::TS {
//                 index_alloc: vec![idx_us, idx_sv],
//                 data: s_ten,
//             },
//             Self::TV {
//                 index_alloc: v_idxv,
//                 data: v,
//             },
//         ))
//     }
// }

impl ElementAccess for NdDenseRepr<f64> {
    type Index = usize;
    type E = f64;
    type Err = NdDenseReprError;

    fn get(
        &self,
        locs: impl IntoIterator<Item = impl Borrow<Self::Index>>,
    ) -> Result<&Self::E, Self::Err> {
        let locv: Vec<usize> = locs.into_iter().map(|i| *i.borrow()).collect();
        self.data.get(locv.as_slice()).ok_or(NdDenseReprError)
    }

    fn get_mut(
        &mut self,
        locs: impl IntoIterator<Item = impl Borrow<Self::Index>>,
    ) -> Result<&mut Self::E, Self::Err> {
        let locv: Vec<usize> = locs.into_iter().map(|i| *i.borrow()).collect();
        self.data.get_mut(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

pub type NdDenseTensor<Id, E> = Tensor<Id, NdDenseRepr<E>>;
impl<E, Id: LegId> NdDenseTensor<Id, E> {
    pub fn random(leg_size_set: impl IntoIterator<Item = LegVal<Id, usize>>) -> Self
    where
        E: Scalar,
    {
        let (id_iter, size_iter): (Vec<_>, Vec<_>) = leg_size_set
            .into_iter()
            .map(|LegVal(id, s)| (id, s))
            .unzip();
        Self::from_raw(NdDenseRepr::random(size_iter), id_iter)
            .map_err(|_| "s")
            .unwrap()
    }
    pub fn zero(leg_size_set: impl IntoIterator<Item = LegVal<Id, usize>>) -> Self
    where
        E: Clone + Zero,
    {
        let (id_iter, size_iter): (Vec<_>, Vec<_>) = leg_size_set
            .into_iter()
            .map(|LegVal(id, s)| (id, s))
            .unzip();
        Self::from_raw(NdDenseRepr::zero(size_iter), id_iter)
            .map_err(|_| "why!?")
            .unwrap()
    }
    pub fn view(&self) -> Tensor<Id, NdDenseViewRepr<E>> {
        Tensor::from_raw_and_bimap(self.raw().view(), self.legs().clone())
            .map_err(|_| "why!?")
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use num_traits::abs;

    use crate::{
        basic::leg::{Id128, Prime},
        rv, v,
    };

    use super::*;

    use std::{collections::HashSet, ops::Not};

    const EPS: f64 = 1e-8;

    // #[test]
    // fn index_mul_test() {
    //     let a = Index::new(3);
    //     let b = Index::new(4);
    //     let c = Index::new(5);
    //     let d = Index::new(6);
    //     let i = Index::new(7);
    //     let j = Index::new(8);
    //     let x_idxv = vec![a, i.clone(), b, j.clone()];
    //     let y_idxv = vec![j, c, d, i];

    //     let x_dims: Vec<usize> = x_idxv.iter().map(|idx| idx.dim()).collect();
    //     let y_dims: Vec<usize> = y_idxv.iter().map(|idx| idx.dim()).collect();
    //     let mut x = ArrayD::<f64>::zeros(IxDyn(&x_dims));
    //     let mut y: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<ndarray::IxDynImpl>> =
    //         ArrayD::<f64>::zeros(IxDyn(&y_dims));

    //     let x_idxs: HashSet<Index> = x_idxv.iter().cloned().collect();
    //     let y_idxs: HashSet<Index> = y_idxv.iter().cloned().collect();
    //     let conn_idxs: HashSet<Index> = x_idxs.intersection(&y_idxs).cloned().collect();
    //     let x_rem_idxs = x_idxs.difference(&y_idxs);
    //     let y_rem_idxs = y_idxs.difference(&y_idxs);
    //     let rem_idxs = x_idxs.symmetric_difference(&y_idxs);

    //     let mut x_idxv_marked: Vec<(bool, Index)> = x_idxv
    //         .iter()
    //         .cloned()
    //         .map(|idx| (conn_idxs.contains(&idx), idx))
    //         .collect();
    //     x_idxv_marked.sort();
    //     let x_idxv_ordered: Vec<Index> = x_idxv_marked.into_iter().map(|(_, idx)| idx).collect();

    //     let mut y_idxv_marked: Vec<(bool, Index)> = y_idxv
    //         .iter()
    //         .cloned()
    //         .map(|idx| (conn_idxs.contains(&idx).not(), idx))
    //         .collect();
    //     y_idxv_marked.sort();
    //     let y_idxv_ordered: Vec<Index> = y_idxv_marked.into_iter().map(|(_, idx)| idx).collect();

    //     x_idxv.iter().for_each(|idx| println!("{}", idx));
    //     x_idxv_ordered.iter().for_each(|idx| println!("{}", idx));
    //     y_idxv.iter().for_each(|idx| println!("{}", idx));
    //     y_idxv_ordered.iter().for_each(|idx| println!("{}", idx));

    //     let mut x_idxv_to_be_ordered = x_idxv;
    //     let mut y_idxv_to_be_ordered = y_idxv;

    //     for i in 0..x_idxv_to_be_ordered.len() {
    //         for j in i..x_idxv_to_be_ordered.len() {
    //             if x_idxv_to_be_ordered[j] == x_idxv_ordered[i] {
    //                 x_idxv_to_be_ordered.swap(i, j);
    //                 x.swap_axes(i, j);
    //             }
    //         }
    //     }

    //     for i in 0..y_idxv_to_be_ordered.len() {
    //         for j in i..y_idxv_to_be_ordered.len() {
    //             if y_idxv_to_be_ordered[j] == y_idxv_ordered[i] {
    //                 y_idxv_to_be_ordered.swap(i, j);
    //                 y.swap_axes(i, j);
    //             }
    //         }
    //     }

    //     assert_eq!(x_idxv_to_be_ordered, x_idxv_ordered);
    //     assert_eq!(y_idxv_to_be_ordered, y_idxv_ordered);

    //     println!("{:?} vs {:?}", x.shape(), y.shape());
    // }

    type Index = Prime<Id128>;

    #[test]
    fn tensor_mul_test() -> Result<(), anyhow::Error> {
        let a_n = 3;
        let b_n = 4;
        let c_n = 5;
        let d_n = 6;
        let i_n = 7;
        let j_n = 8;

        let a = Index::new();
        let b = Index::new();
        let c = Index::new();
        let d = Index::new();
        let i = Index::new();
        let j = Index::new();

        let ta = NdDenseTensor::<_, f64>::random(v![a=>a_n, i=>i_n, b=>b_n, j=>j_n]);
        let tb = NdDenseTensor::<_, f64>::random(v![j=>j_n, c=>c_n, d=>d_n, i=>i_n]);

        let alv = ta.legs();
        let blv = tb.legs();
        println!("{:?}", alv);
        println!("{:?}", blv);

        //println!("{:?}", tc.index_set());

        let tc = (ta.view() * tb.view()).by(())?;

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
                                tc_e += ta.get(rv![a=>ai, i=>ii, b=>bi, j=>ji])?
                                    * tb.get(rv![j=>ji, c=>ci, d=>di, i=>ii])?;
                            }
                        }
                        let tc_r = tc.get(rv![a=>ai,b=>bi,c=>ci,d=>di])?;
                        //tcd[[ai, bi, ci, di]];
                        println!("{},{},{},{} : {} vs {}", ai, bi, ci, di, tc_e, tc_r);
                        assert!(abs(tc_e - tc_r) < EPS);
                    }
                }
            }
        }

        let clv = tc.legs();
        println!("{:?}", clv);

        Ok(())
    }

    // #[test]
    // fn tensor_svd_test() -> Result<()> {
    //     let a = Index::new(3);
    //     let b = Index::new(4);
    //     let c = Index::new(5);
    //     let d = Index::new(6);
    //     let t = DenseTensor::<f64>::random(&[a, b, c, d]);

    //     let (u, s, v) = (&t).svd(&[a, b], ())?;
    //     let usv = ((&u * &s)? * &v)?;

    //     for ai in 0..3 {
    //         for bi in 0..4 {
    //             for ci in 0..5 {
    //                 for di in 0..6 {
    //                     assert!(
    //                         abs(t.get(&loc![a=>ai,b=>bi,c=>ci,d=>di])?
    //                             - usv.get(&loc![a=>ai,b=>bi,c=>ci,d=>di])?)
    //                             < EPS
    //                     );
    //                 }
    //             }
    //         }
    //     }

    //     println!("ut");
    //     let mut ut = (!(&u))?;
    //     ut.replace_index(a, a.prime())?;
    //     ut.replace_index(b, b.prime())?;
    //     println!("{:?}", ut.index_set());
    //     let uut = (u * ut)?;
    //     println!("{:?}", uut.index_set());
    //     for ai in 0..3 {
    //         for bi in 0..4 {
    //             for api in 0..3 {
    //                 for bpi in 0..4 {
    //                     let re = *uut.get(&loc![a=>ai,b=>bi,a.prime()=>api,b.prime()=>bpi])?;
    //                     if ai == api && bi == bpi {
    //                         assert!(abs(re - 1.) < EPS);
    //                     } else {
    //                         assert!(abs(re) < EPS);
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     Ok(())
    // }
}
