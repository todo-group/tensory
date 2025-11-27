use core::convert::Infallible;

use alloc::vec::Vec;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ndarray_linalg::{Lapack, Norm, Scalar, from_diag};
use num_traits::ConstZero;
use tensory_core::mapper::{EquivGroupedAxes, GroupedAxes};
use tensory_linalg::{
    conj::ConjCtx,
    eig::EigCtxImpl,
    exp::ExpCtxImpl,
    norm::{NormCtx, NormRuntime},
    pow::PowCtxImpl,
    qr::QrCtxImpl,
    solve_eig::SolveEigCtxImpl,
    svd::{SvdCtxImpl, SvdWithOptionRuntime},
};

use crate::{
    NdDenseRepr, NdDenseViewRepr, NdRuntime,
    cut_filter::CutFilter,
    tenalg::{conj, diag_map, error::TenalgError, into_eig, into_eigh, into_qr, into_svddc},
};

unsafe impl<'a, E: Scalar + Lapack, C: CutFilter<<E as Scalar>::Real>>
    SvdCtxImpl<NdDenseViewRepr<'a, E>> for (C,)
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

impl<'a, E: Scalar + Lapack, C: CutFilter<<E as Scalar>::Real>>
    SvdWithOptionRuntime<NdDenseViewRepr<'a, E>, C> for NdRuntime
where
    E::Real: ConstZero,
{
    type Ctx = (C,);

    fn svd_ctx(&self, opt: C) -> Self::Ctx {
        (opt,)
    }
}

unsafe impl<'a, E: Scalar + Lapack> QrCtxImpl<NdDenseViewRepr<'a, E>> for () {
    type Q = NdDenseRepr<E>;
    type R = NdDenseRepr<E>;
    type Err = TenalgError;

    unsafe fn qr_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: GroupedAxes<2>,
    ) -> Result<(Self::Q, Self::R), Self::Err> {
        let (_, [q_set, r_set]) = axes_split.into_raw();
        // U 0 uset
        // S 0 1
        // V 1 vset

        let a_raw = a.data;

        let q_set_len = q_set.len();

        let a_idxv_ordered: Vec<usize> =
            q_set.iter().cloned().chain(r_set.iter().cloned()).collect();

        let a_rot = a_raw.permuted_axes(a_idxv_ordered);

        let (q, r) = into_qr(a_rot, q_set_len)?;

        let q = q.permuted_axes(
            core::iter::once(q_set_len)
                .chain(0..q_set_len)
                .collect::<Vec<_>>(),
        );

        Ok((NdDenseRepr { data: q }, NdDenseRepr { data: r }))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct HermiteEig;

unsafe impl<'a, E: Scalar + Lapack> EigCtxImpl<NdDenseViewRepr<'a, E>> for HermiteEig {
    type V = NdDenseRepr<E>;
    type D = NdDenseRepr<E::Real>;
    type VC = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn eig_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: tensory_core::mapper::EquivGroupedAxes<2>,
    ) -> Result<(Self::V, Self::D, Self::VC), Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // VC 0 uset
        // D  0 1
        // V  1 vset

        let a_raw = a.data;

        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();

        let a_rot = a_raw.permuted_axes(a_idxv_ordered);

        let (vc, d) = into_eigh(a_rot, l_set_len)?;

        let d_ten = from_diag(&d.to_vec()).into_dimensionality::<IxDyn>()?;

        let vc = vc.permuted_axes(
            core::iter::once(l_set_len)
                .chain(0..l_set_len)
                .collect::<Vec<_>>(),
        );

        let v = vc.map(|e| e.conj());

        Ok((
            NdDenseRepr { data: vc },
            NdDenseRepr { data: d_ten },
            NdDenseRepr { data: v },
        ))
    }
}

unsafe impl<'a, E: Scalar + Lapack> SolveEigCtxImpl<NdDenseViewRepr<'a, E>> for () {
    type D = NdDenseRepr<E::Complex>;
    type V = NdDenseRepr<E::Complex>;

    type Err = TenalgError;

    unsafe fn solve_eig_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: tensory_core::mapper::EquivGroupedAxes<2>,
    ) -> Result<(Self::D, Self::V), Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // D  0 1
        // V  1 vset

        let a_raw = a.data;

        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();

        let a_rot = a_raw.permuted_axes(a_idxv_ordered);

        let (v, d) = into_eig(a_rot, l_set_len)?;

        let d_ten = from_diag(&d.to_vec()).into_dimensionality::<IxDyn>()?;

        let v = v.permuted_axes(
            core::iter::once(l_set_len)
                .chain(0..l_set_len)
                .collect::<Vec<_>>(),
        );

        Ok((NdDenseRepr { data: d_ten }, NdDenseRepr { data: v }))
    }
}

impl<E: Scalar + Lapack> NormCtx<NdDenseViewRepr<'_, E>> for () {
    type Res = E::Real;
    type Err = Infallible;

    fn norm(self, a: NdDenseViewRepr<'_, E>) -> core::result::Result<Self::Res, Self::Err> {
        Ok(a.data.norm_l2())
    }
}
impl<'a, E: Scalar + Lapack> NormRuntime<NdDenseViewRepr<'a, E>> for NdRuntime {
    type Ctx = ();

    fn norm_ctx(&self) -> Self::Ctx {}
}

unsafe impl<'a, E: Scalar> ConjCtx<NdDenseViewRepr<'a, E>> for () {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    fn conjugate(self, a: NdDenseViewRepr<'a, E>) -> Result<Self::Res, Self::Err> {
        Ok(NdDenseRepr {
            data: conj(&a.data)?,
        })
    }
}

pub struct DiagExp;

unsafe impl<'a, E: Scalar> ExpCtxImpl<NdDenseViewRepr<'a, E>> for DiagExp {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn exp_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // lset rset

        let a_raw = a.data;
        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a_raw.permuted_axes(a_idxv_ordered);

        #[cfg(test)]
        {
            std::println!("a_rot shape: {:?}", a_rot.shape());
            std::println!("l_set rset: {:?} {:?}", l_set, r_set);
        }

        let raw = diag_map(a_rot, l_set_len, |x| *x = x.exp())?;
        Ok(NdDenseRepr { data: raw })
    }
}

pub struct DiagPow;
pub struct DiagPowF;
//pub struct DiagPowC;

unsafe impl<'a, E: Scalar> PowCtxImpl<NdDenseViewRepr<'a, E>, E> for DiagPow {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn pow_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        power: E,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // lset rset

        let a_raw = a.data;
        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a_raw.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.pow(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}
unsafe impl<'a, E: Scalar> PowCtxImpl<NdDenseViewRepr<'a, E>, E::Real> for DiagPowF {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn pow_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        power: E::Real,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // lset rset

        let a_raw = a.data;
        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a_raw.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.powf(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}
// unsafe impl<'a, E: Scalar> PowCtxImpl<NdDenseViewRepr<'a, E>, E::Complex> for DiagPowC {
//     type Res = NdDenseRepr<E>;

//     type Err = TenalgError;

//     unsafe fn pow_unchecked(
//         self,
//         a: NdDenseViewRepr<'a, E>,
//         power: E::Real,
//         axes_split: EquivGroupedAxes<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         let (_, [l_set, r_set]) = axes_split.into_raw();
//         // lset rset

//         let a_raw = a.data;
//         let l_set_len = l_set.len();

//         let a_idxv_ordered: Vec<usize> =
//             l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
//         let a_rot = a_raw.permuted_axes(a_idxv_ordered);
//         let raw = diag_map(a_rot, l_set_len, |x| *x = x.powf(power))?;
//         Ok(NdDenseRepr { data: raw })
//     }
// }

pub struct DiagPowI;

unsafe impl<'a, E: Scalar> PowCtxImpl<NdDenseViewRepr<'a, E>, i32> for DiagPowI {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn pow_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        power: i32,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // lset rset

        let a_raw = a.data;
        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a_raw.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.powi(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}

pub struct Half;

unsafe impl<'a, E: Scalar> PowCtxImpl<NdDenseViewRepr<'a, E>, Half> for () {
    type Res = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn pow_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        _power: Half,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (_, [l_set, r_set]) = axes_split.into_raw();
        // lset rset

        let a_raw = a.data;
        let l_set_len = l_set.len();

        let a_idxv_ordered: Vec<usize> =
            l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a_raw.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.sqrt())?;
        Ok(NdDenseRepr { data: raw })
    }
}

#[cfg(test)]
mod tests {

    use std::{println, vec};

    use anyhow::Ok;
    use ndarray::array;
    use ndarray_linalg::Scalar;
    use tensory_basic::{
        id::{Id128, Prime},
        mapper::VecMapper,
    };
    use tensory_core::prelude::*;
    use tensory_linalg::{
        eig::TensorEigExt, exp::TensorExpExt, qr::TensorQrExt, solve_eig::TensorSolveEigExt,
        svd::TensorSvdExt,
    };

    use crate::{
        NdDenseRepr, NdDenseTensor, NdDenseTensorExt,
        linalg::{DiagExp, HermiteEig},
    };
    use tensory_core::tensor::TensorTask;

    type Leg = Prime<Id128>;

    type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;

    const EPS: f64 = 1e-8;

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

        let t = Tensor::random(lm![a=>a_n, b=>b_n, c=>c_n, d=>d_n]).unwrap();

        let us = Leg::new();
        let vs = Leg::new();

        let (u, s, v) = t.view().svd(ls![&a, &b], us, vs)?.with(((),))?;

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
                            (t.get(lm![&a=>ai,&b=>bi,&c=>ci,&d=>di])??
                                - usv.get(lm![&a=>ai,&b=>bi,&c=>ci,&d=>di])??)
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

        let ut = ut.replace_leg(lm![&a=> a.prime(), &b=> b.prime()]).unwrap();
        println!("{:?}", ut.mapper());
        let uut = (&u * ut)?.with(())?;
        println!("{:?}", uut.mapper());
        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        let re = *uut.get(lm![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
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

    #[test]
    fn tensor_qr_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        let c = Leg::new();
        let d = Leg::new();
        let a_n = 10;
        let b_n = 15;
        let c_n = 20;
        let d_n = 25;

        let t = Tensor::random(lm![a=>a_n, c=>c_n, d=>d_n, b=>b_n]).unwrap();

        let qr_leg = Leg::new();

        let (q, r) = (&t).qr(ls![&a, &b], qr_leg)?.with(())?;

        //let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        println!("{:?}\n{:?}\n{:?}\n{:?}\n{:?}\n", a, b, c, d, qr_leg);

        println!("{:?}\n{:?}\n", q.mapper(), r.mapper());

        println!("{:?} {:?}", q.repr().data.shape(), r.repr().data.shape());

        let qr = (&q * &r)?.with(())?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                for ci in 0..c_n {
                    for di in 0..d_n {
                        assert!(
                            (t.get(lm![&a=>ai,&b=>bi,&c=>ci,&d=>di])??
                                - qr.get(lm![&a=>ai,&b=>bi,&c=>ci,&d=>di])??)
                            .abs()
                                < EPS
                        );
                    }
                }
            }
        }

        // println!("ut");
        let qt = q.view();

        let ap = a.prime();
        let bp = b.prime();

        println!("{:?} {:?} {:?}:  {:?}", a, b, qr_leg, qt.mapper());

        let qt = qt.replace_leg(lm![&a=> a.prime(), &b=> b.prime()]).unwrap();
        println!("{:?}", qt.mapper());
        let qqt = (&q * qt)?.with(())?;
        println!("{:?}", qqt.mapper());
        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        let re = *qqt.get(lm![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
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

    #[test]
    fn tensor_eigh_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        //let c = Leg::new();
        let a_n = 10;
        let b_n = 15;
        //let c_n = 20;

        let t = Tensor::random_hermite(lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n])?;

        // let t = Tensor::from_raw(
        //     NdDenseRepr {
        //         data: ndarray_linalg::random_hermite(a_n * b_n)
        //             .into_shape_with_order(vec![a_n, b_n, a_n, b_n])?,
        //     },
        //     VecMapper::build([a, b, a.prime(), b.prime()].into_iter())?,
        // )
        // .unwrap();

        let vcd = Leg::new();
        let dv = Leg::new();

        let (vc, d, v) = t
            .view()
            .eig(ls![(&a, &a.prime()), (&b, &b.prime())], vcd, dv)?
            .with(HermiteEig)?;

        //let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        println!("{:?}\n{:?}\n{:?}\n{:?}\n", a, b, vcd, dv);

        println!("{:?}\n{:?}\n{:?}\n", vc.mapper(), d.mapper(), v.mapper());

        println!(
            "{:?} {:?} {:?}",
            vc.repr().data.shape(),
            d.repr().data.shape(),
            v.repr().data.shape()
        );

        let vcd_tmp = (&vc * &d)?.with(())?;

        println!("{:?} {:?}", vcd_tmp.mapper(), v.repr().data.shape());

        let usv = (vcd_tmp.view() * v.view())?.with(())?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        assert!(
                            (t.get(lm![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi])??
                                - usv
                                    .get(lm![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi])??)
                            .abs()
                                < EPS
                        );
                    }
                }
            }
        }

        // // println!("ut");
        // let ut = u.view();

        // let ap = a.prime();
        // let bp = b.prime();

        // println!("{:?} {:?} {:?}:  {:?}", a, b, us, ut.mapper());

        // let ut = ut.replace_leg(&a, a.prime()).unwrap();
        // let ut = ut.replace_leg(&b, b.prime()).unwrap();
        // println!("{:?}", ut.mapper());
        // let uut = (&u * ut)?.with(())?;
        // println!("{:?}", uut.mapper());
        // for ai in 0..a_n {
        //     for bi in 0..b_n {
        //         for api in 0..a_n {
        //             for bpi in 0..b_n {
        //                 let re = *uut.get(lm![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
        //                 if ai == api && bi == bpi {
        //                     assert!((re - 1.).abs() < EPS);
        //                 } else {
        //                     assert!(re.abs() < EPS);
        //                 }
        //             }
        //         }
        //     }
        // }

        Ok(())
    }

    #[test]
    fn tensor_eig_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        //let c = Leg::new();
        let a_n = 10;
        let b_n = 20;
        //let c_n = 20;

        let t = Tensor::random(lm![b=>b_n,a.prime()=>a_n,a=>a_n,b.prime()=>b_n])?;

        let vd = Leg::new();
        let dvinv = Leg::new();

        let (v, d) = t
            .view()
            .solve_eig(ls![(&a, &a.prime()), (&b, &b.prime())], vd, dvinv)?
            .with(())?;

        //let s = s.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        println!("{:?}\n{:?}\n{:?}\n{:?}\n", a, b, vd, dvinv);

        println!("{:?}\n{:?}\n", v.mapper(), d.mapper(),);

        println!("{:?} {:?}", v.repr().data.shape(), d.repr().data.shape());

        let vd_tmp = (&v * &d)?.with(())?;

        let t_comp = t.map(|e| <f64 as Scalar>::Complex::from_real(*e));

        let tv_tmp = (&t_comp
            * v.view()
                .replace_leg(lm![&a => a.prime(), &b => b.prime(),&vd => dvinv])?)?
        .with(())?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                for di in 0..a_n * b_n {
                    assert!(
                        (vd_tmp.get(lm![&a=>ai,&b=>bi,&dvinv=>di])??
                            - tv_tmp.get(lm![&a=>ai,&b=>bi,&dvinv=>di])??)
                        .abs()
                            < EPS
                    );
                }
            }
        }

        Ok(())
    }

    #[test]
    fn tensor_diag_exp_test() -> anyhow::Result<()> {
        let a = Leg::new();
        let b = Leg::new();
        //let c = Leg::new();
        let a_n = 10;
        let b_n = 20;
        //let c_n = 20;

        let mut t = Tensor::zero(lm![a=>a_n, a.prime()=>a_n, b=>b_n, b.prime()=>b_n])?;
        let t = Tensor::eye(lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n])?;

        println!("{:?}", t.repr().data.shape());

        // for ai in 0..a_n {
        //     for bi in 0..b_n {
        //         t[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]] = (ai + bi) as f64;
        //     }
        // }

        let t_exp = (&t)
            .exp(ls![(&a, &a.prime()), (&b, &b.prime())])?
            .with(DiagExp); // diagonal exp

        {
            let t = Tensor::eye(lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n])?; // [a,a',b,b']
            //let t = Tensor::<f64>::zero(lm![a=>a_n,a.prime()=>a_n,b=>b_n,b.prime()=>b_n])?; // [a,a',b,b']

            println!("{:?}", t.repr());

            let n = (&t)
                .exp(ls![(&a, &a.prime()), (&b, &b.prime())])?
                .with(DiagExp)?; // scalar
        }

        Ok(())
    }
}
