#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

extern crate blas_src;

mod tenalg;

pub mod cut_filter {
    pub use crate::tenalg::cut_filter::*;
}

use core::{
    borrow::Borrow,
    ops::{Div, Mul},
    panic,
};

use tensory_core::{
    args::LegMapArg,
    arith::{
        CommutativeScalarDivContext, CommutativeScalarMulContext, ConjugationContext, MulRuntime,
    },
    mapper::{AxisMapper, BuildableMapper, ConnectAxisOrigin, GroupedAxes},
    repr::{AsViewMutRepr, AsViewRepr},
    tensor::Tensor,
    utils::elem_get::{ElemGetMutReprImpl, ElemGetReprImpl},
};

use alloc::vec::Vec;
use ndarray::{ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, IxDyn, OwnedRepr, ScalarOperand};
use ndarray_linalg::{Lapack, Scalar, from_diag, random};
use num_traits::{ConstZero, Zero};
use tensory_core::{arith::MulCtxImpl, repr::TensorRepr};
use tensory_linalg::svd::SvdContextImpl;

use crate::tenalg::{conj, cut_filter::CutFilter, error::TenalgError, into_svd, into_svddc, mul};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseRepr<E> {
    data: ArrayD<E>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseViewRepr<'a, E> {
    data: ArrayViewD<'a, E>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct NdDenseViewMutRepr<'a, E> {
    data: ArrayViewMutD<'a, E>,
}

impl<E> NdDenseRepr<E> {
    fn random(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        Self {
            data: random(sizes),
        }
    }
    fn zero(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Clone + Zero,
    {
        let sizes: Vec<usize> = sizes.collect();
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

unsafe impl<'a, E: 'a> AsViewRepr<'a> for NdDenseRepr<E> {
    type View = NdDenseViewRepr<'a, E>;
    fn view(&'a self) -> Self::View {
        NdDenseViewRepr {
            data: self.data.view(),
        }
    }
}
unsafe impl<'a, E: 'a> AsViewMutRepr<'a> for NdDenseRepr<E> {
    type ViewMut = NdDenseViewMutRepr<'a, E>;
    fn view_mut(&'a mut self) -> Self::ViewMut {
        NdDenseViewMutRepr {
            data: self.data.view_mut(),
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
unsafe impl<E> TensorRepr for NdDenseViewMutRepr<'_, E> {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdDenseReprError;
impl core::error::Error for NdDenseReprError {}
impl core::fmt::Display for NdDenseReprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DenseRepr operation failed")
    }
}

unsafe impl<E: ScalarOperand + Mul<Output = E> + Clone>
    CommutativeScalarMulContext<NdDenseRepr<E>, E> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    fn scalar_mul(self, a: NdDenseRepr<E>, scalar: E) -> Result<Self::Res, Self::Err> {
        Ok(NdDenseRepr {
            data: a.data * scalar,
        })
    }
}

unsafe impl<E: ScalarOperand + Div<Output = E> + Clone>
    CommutativeScalarDivContext<NdDenseRepr<E>, E> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    fn scalar_div(self, a: NdDenseRepr<E>, scalar: E) -> Result<Self::Res, Self::Err> {
        Ok(NdDenseRepr {
            data: a.data / scalar,
        })
    }
}

//struct NdDenseContext;

unsafe impl<'l, 'r, E: Lapack + Scalar + ConstZero>
    MulCtxImpl<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>> for ()
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    unsafe fn mul_unchecked(
        self,
        lhs: NdDenseViewRepr<'l, E>,
        rhs: NdDenseViewRepr<'r, E>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        let ([lhs_dim, rhs_dim], contr_pairs) = axis_origin.into_raw();

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

        // let lhs_rem_idxa = &lhs_idxv_ordered.as_slice()[0..lhs_idxv_ordered.len() - conn_idxn];
        // let rhs_rem_idxa = &rhs_idxv_ordered.as_slice()[conn_idxn..];

        #[cfg(test)]
        {
            use std::println;
            println!("{:?}", lhs_rot.dim());
            println!("{:?}", lhs_idxv_ordered);
            println!("{:?}", rhs_rot.dim());
            println!("{:?}", rhs_idxv_ordered);

            //println!("{:?}", lhs_rem_idxa);
            //println!("{:?}", rhs_rem_idxa);
        }

        // let mut z_idxv: Vec<(usize, usize)> = (0..z.shape().len()).map(|i| (0, i)).collect();

        // res_origin.into_iter().enumerate().for_each(|(k, (t, i))| {
        //     z_idxv[match t {
        //         0 => lhs_rem_idxa.iter().position(|&j| i == j).unwrap(),
        //         1 => rhs_rem_idxa.iter().position(|&j| i == j).unwrap() + lhs_rem_idxa.len(),
        //         _ => panic!("Unexpected axis"),
        //     }]
        //     .0 = k;
        // });

        // z_idxv.sort();

        // let z_idxv = z_idxv.into_iter().map(|(_, i)| i).collect::<Vec<_>>();

        //let z = z.permuted_axes(z_idxv);

        // let z_idxv = x_rem_idxa
        //     .iter()
        //     .map(|idx| MulAxisOrigin::Lhs(*idx))
        //     .chain(y_rem_idxa.iter().map(|idx| MulAxisOrigin::Rhs(*idx)))
        //     .collect();

        Ok(NdDenseRepr { data: z })
    }
}

unsafe impl<'a, E: Scalar + Lapack, C: CutFilter<<E as Scalar>::Real>>
    SvdContextImpl<NdDenseViewRepr<'a, E>> for (C,)
where
    E::Real: ConstZero,
{
    type U = NdDenseRepr<E>;
    type S = NdDenseRepr<E::Real>;
    type V = NdDenseRepr<E>;
    type Err = TenalgError;

    unsafe fn svd_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err> {
        let (_, [u_set, v_set]) = axes_split.into_raw();
        // U 0 uset
        // S 0 1
        // V 1 vset

        let a_raw = a.data;

        let u_set_len = u_set.len();

        let a_idxv_ordered: Vec<usize> =
            u_set.iter().cloned().chain(v_set.iter().cloned()).collect();

        let a_rot = a_raw.permuted_axes(a_idxv_ordered);

        let (u, s, v) = into_svddc(a_rot, u_set_len, self.0)?;

        let u = u.permuted_axes(
            core::iter::once(u_set_len)
                .chain(0..u_set_len)
                .collect::<Vec<_>>(),
        );

        let s: ArrayBase<OwnedRepr<<E as Scalar>::Real>, ndarray::Dim<[usize; 1]>> = s;

        let s_ten = from_diag(&s.to_vec()).into_dimensionality::<IxDyn>()?;

        Ok((
            NdDenseRepr { data: u },
            NdDenseRepr { data: s_ten },
            NdDenseRepr { data: v },
        ))
    }
}

impl<E> ElemGetReprImpl for NdDenseRepr<E> {
    type Index = usize;
    type E = E;
    type Err = NdDenseReprError;

    unsafe fn get_unchecked(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

impl<E> ElemGetMutReprImpl for NdDenseRepr<E> {
    unsafe fn get_mut_unchecked(
        &mut self,
        indices: Vec<Self::Index>,
    ) -> Result<&mut Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get_mut(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

unsafe impl<'a, E: Scalar> ConjugationContext<NdDenseViewRepr<'a, E>> for () {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    fn conjugate(self, a: NdDenseViewRepr<'a, E>) -> Result<Self::Res, Self::Err> {
        Ok(NdDenseRepr {
            data: conj(&a.data)?,
        })
    }
}
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

pub type NdDenseTensor<E, B> = Tensor<NdDenseRepr<E>, B>;

pub trait NdDenseTensorExt<E, B: AxisMapper>: Sized {
    fn zero<
        K: ExactSizeIterator + Iterator<Item = B::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <B as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        B: BuildableMapper<K>;
    fn random<
        K: ExactSizeIterator + Iterator<Item = B::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <B as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        B: BuildableMapper<K>;
    fn map<E2, F: FnMut(&E) -> E2>(&self, f: F) -> Tensor<NdDenseRepr<E2>, B>
    where
        B: Clone;
}
impl<E, B: AxisMapper> NdDenseTensorExt<E, B> for NdDenseTensor<E, B> {
    fn zero<
        K: ExactSizeIterator + Iterator<Item = B::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <B as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        B: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();

        let broker = B::build(k)?;

        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::zero(v), broker) })
    }
    fn random<
        K: ExactSizeIterator + Iterator<Item = B::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <B as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        B: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let broker = B::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random(v), broker) })
    }
    fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> Tensor<NdDenseRepr<E2>, B>
    where
        B: Clone,
    {
        unsafe { Tensor::from_raw_unchecked(self.repr().map(&mut f), self.mapper().clone()) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdRuntime;
impl<'l, 'r, E: Scalar + Lapack + ConstZero>
    MulRuntime<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>> for &NdRuntime
{
    type Ctx = ();
    fn mul_ctx(self) -> Self::Ctx {}
}

#[cfg(test)]
mod tests {
    use std::println;

    use num_traits::abs;
    use tensory_core::leg;
    use tensory_linalg::svd::TensorSvdExt;

    use super::*;

    use tensory_basic::{
        broker::VecBroker,
        id::{Id128, Prime},
    };

    const EPS: f64 = 1e-8;

    type Leg = Prime<Id128>;

    type Tensor = NdDenseTensor<f64, VecBroker<Leg>>;

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

        let ta = Tensor::random(leg![a => a_n, i => i_n, b => b_n, j => j_n]).unwrap();
        let tb = Tensor::random(leg![j => j_n, c => c_n, d => d_n, i => i_n]).unwrap();

        let nd = NdRuntime;

        let alv = ta.mapper();
        let blv = tb.mapper();
        println!("{:?}", alv);
        println!("{:?}", blv);

        let ta = ta.bind(&nd);
        let tb = tb.bind(&nd);

        //println!("{:?}", tc.leg_alloc());

        let tc = (&ta * &tb)?;

        let ta = ta.unbind();
        let tb = tb.unbind();

        let tc = tc.unbind();

        let tc = (tc * (10.0,)).with(())?;

        let tc = (tc / (10.0,)).with(())?;

        let clv = tc.mapper();
        println!("{:?}", clv);

        // let (ta, _) = ta.into_raw();
        // let (tb, _) = tb.into_raw();
        // let (tc, _) = tc.into_raw();

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
                                tc_e += ta.get(leg![&a=>ai, &i=>ii, &b=>bi, &j=>ji])??
                                    * tb.get(leg![&j=>ji, &c=>ci, &d=>di, &i=>ii])??;
                            }
                        }
                        let tc_r = tc.get(leg![&a=>ai, &b=>bi, &c=>ci, &d=>di])??;
                        //tcd[[ai, bi, ci, di]];
                        println!("{},{},{},{} : {} vs {}", ai, bi, ci, di, tc_e, tc_r);
                        assert!(abs(tc_e - tc_r) < EPS);
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn tensor_svd_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        let c = Leg::new();
        let d = Leg::new();
        let a_n = 10;
        let b_n = 15;
        let c_n = 20;
        let d_n = 25;

        let t = Tensor::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n]).unwrap();

        let us = Leg::new();
        let vs = Leg::new();

        let (u, s, v) = t.view().svd(leg![&a, &b], us, vs)?.with(((),))?;

        //let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        println!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", a, b, c, d, us, vs);

        println!("{:?}\n{:?}\n{:?}\n", u.mapper(), s.mapper(), v.mapper());

        println!(
            "{:?} {:?} {:?}",
            u.repr().data.shape(),
            s.repr().data.shape(),
            v.repr().data.shape()
        );

        let us_tmp = (&u * &s)?.with(())?;

        println!("{:?} {:?}", us_tmp.mapper(), u.repr().data.shape());

        let usv = (us_tmp.view() * v.view())?.with(())?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                for ci in 0..c_n {
                    for di in 0..d_n {
                        assert!(
                            (t.get(leg![&a=>ai,&b=>bi,&c=>ci,&d=>di])??
                                - usv.get(leg![&a=>ai,&b=>bi,&c=>ci,&d=>di])??)
                            .abs()
                                < EPS
                        );
                    }
                }
            }
        }

        // println!("ut");
        let ut = u.view();

        let ap = a.prime();
        let bp = b.prime();

        println!("{:?} {:?} {:?}:  {:?}", a, b, us, ut.mapper());

        let ut = ut.replace_leg(&a, a.prime()).unwrap();
        let ut = ut.replace_leg(&b, b.prime()).unwrap();
        println!("{:?}", ut.mapper());
        let uut = (&u * ut)?.with(())?;
        println!("{:?}", uut.mapper());
        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        let re = *uut.get(leg![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
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
