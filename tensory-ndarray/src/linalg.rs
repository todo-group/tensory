use core::convert::Infallible;

use alloc::vec::Vec;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ndarray_linalg::{Lapack, Norm, Scalar, from_diag};
use num_traits::ConstZero;
use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, RuntimeFor},
    },
    mapper::{EquivGroupedAxes, GroupedAxes},
    repr::ReprContext,
};
use tensory_linalg::{
    conj::ConjRepr,
    eig::EigRepr,
    exp::ExpRepr,
    norm::NormRepr,
    pow::PowRepr,
    qr::QrRepr,
    solve_eig::SolveEigRepr,
    svd::SvdRepr,
};

use crate::{
    Nd, NdDenseRepr, NdDenseViewRepr, NdRuntime,
    cut_filter::{CutFilter, SingularFilter},
    tenalg::{conj, diag_map, error::TenalgErr, into_eig, into_eigh, into_qr, into_svddc},
};

fn execute_svd<
    'a,
    E: Scalar + Lapack,
    F: SingularFilter<OwnedRepr<E>, OwnedRepr<E::Real>, OwnedRepr<E>>,
>(
    a: NdDenseViewRepr<'a, E>,
    axes_split: GroupedAxes<2>,
    filter: F,
) -> Result<(NdDenseRepr<E>, NdDenseRepr<E::Real>, NdDenseRepr<E>), TenalgErr>
where
    E::Real: ConstZero,
{
    let (_, [u_set, v_set]) = axes_split.into_raw();
    let u_set_len = u_set.len();
    let a_idxv_ordered: Vec<usize> = u_set.iter().cloned().chain(v_set.iter().cloned()).collect();
    let a_rot = a.data.permuted_axes(a_idxv_ordered);
    let (u, s, v) = into_svddc(a_rot, u_set_len, filter)?;
    let u = u.permuted_axes(
        core::iter::once(u_set_len)
            .chain(0..u_set_len)
            .collect::<Vec<_>>(),
    );
    let s: ArrayBase<OwnedRepr<E::Real>, ndarray::Dim<[usize; 1]>> = s;
    let s_ten = from_diag(&s.to_vec()).into_dimensionality::<IxDyn>()?;
    Ok((
        NdDenseRepr { data: u },
        NdDenseRepr { data: s_ten },
        NdDenseRepr { data: v },
    ))
}

impl<
        'a,
        E: Scalar + Lapack,
        F: SingularFilter<OwnedRepr<E>, OwnedRepr<E::Real>, OwnedRepr<E>>,
    > Context<Nd, SvdRepr<NdDenseViewRepr<'a, E>, F>> for ()
where
    E::Real: ConstZero,
{
    type Output = Result<(NdDenseRepr<E>, NdDenseRepr<E::Real>, NdDenseRepr<E>), TenalgErr>;

    fn execute(
        self,
        task: SvdRepr<NdDenseViewRepr<'a, E>, F>,
    ) -> Self::Output {
        let (a, axes_split, filter) = task.into_raw();
        execute_svd(a, axes_split, filter)
    }
}

unsafe impl<
        'a,
        E: Scalar + Lapack,
        F: SingularFilter<OwnedRepr<E>, OwnedRepr<E::Real>, OwnedRepr<E>>,
    > ReprContext<Nd, 3, SvdRepr<NdDenseViewRepr<'a, E>, F>> for ()
where
    E::Real: ConstZero,
{
    type Repr = (NdDenseRepr<E>, NdDenseRepr<E::Real>, NdDenseRepr<E>);
    type CType = Resulting<Raw, TenalgErr>;
}

impl<
        'a,
        E: Scalar + Lapack,
        F: SingularFilter<OwnedRepr<E>, OwnedRepr<E::Real>, OwnedRepr<E>>,
    > RuntimeFor<SvdRepr<NdDenseViewRepr<'a, E>, F>> for NdRuntime
where
    E::Real: ConstZero,
{
    type Mk = Nd;
    type Ctx = ();

    fn ctx(&self) -> Self::Ctx {}
}

impl<'a, E: Scalar + Lapack> Context<Nd, QrRepr<NdDenseViewRepr<'a, E>>> for () {
    type Output = Result<(NdDenseRepr<E>, NdDenseRepr<E>), TenalgErr>;

    fn execute(self, task: QrRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        let (a, axes_split) = task.into_raw();
        let (_, [q_set, r_set]) = axes_split.into_raw();
        let q_set_len = q_set.len();
        let a_idxv_ordered: Vec<usize> = q_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let (q, r) = into_qr(a_rot, q_set_len)?;
        let q = q.permuted_axes(
            core::iter::once(q_set_len)
                .chain(0..q_set_len)
                .collect::<Vec<_>>(),
        );
        Ok((NdDenseRepr { data: q }, NdDenseRepr { data: r }))
    }
}

unsafe impl<'a, E: Scalar + Lapack> ReprContext<Nd, 2, QrRepr<NdDenseViewRepr<'a, E>>> for () {
    type Repr = (NdDenseRepr<E>, NdDenseRepr<E>);
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar + Lapack> RuntimeFor<QrRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = ();

    fn ctx(&self) -> Self::Ctx {}
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Hermite;

impl<'a, E: Scalar + Lapack> Context<Nd, EigRepr<NdDenseViewRepr<'a, E>>> for Hermite {
    type Output = Result<
        (NdDenseRepr<E>, NdDenseRepr<E::Real>, NdDenseRepr<E>),
        TenalgErr,
    >;

    fn execute(self, task: EigRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        let (a, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
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

unsafe impl<'a, E: Scalar + Lapack> ReprContext<Nd, 3, EigRepr<NdDenseViewRepr<'a, E>>>
    for Hermite
{
    type Repr = (NdDenseRepr<E>, NdDenseRepr<E::Real>, NdDenseRepr<E>);
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar + Lapack> RuntimeFor<EigRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = Hermite;

    fn ctx(&self) -> Self::Ctx {
        Hermite
    }
}

impl<'a, E: Scalar + Lapack> Context<Nd, SolveEigRepr<NdDenseViewRepr<'a, E>>> for () {
    type Output = Result<
        (NdDenseRepr<E::Complex>, NdDenseRepr<E::Complex>),
        TenalgErr,
    >;

    fn execute(self, task: SolveEigRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        let (a, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let (v, d) = into_eig(a_rot, l_set_len)?;
        let d_ten = from_diag(&d.to_vec()).into_dimensionality::<IxDyn>()?;
        let v = v.permuted_axes(
            core::iter::once(l_set_len)
                .chain(0..l_set_len)
                .collect::<Vec<_>>(),
        );
        Ok((NdDenseRepr { data: v }, NdDenseRepr { data: d_ten }))
    }
}

unsafe impl<'a, E: Scalar + Lapack>
    ReprContext<Nd, 2, SolveEigRepr<NdDenseViewRepr<'a, E>>> for ()
{
    type Repr = (NdDenseRepr<E::Complex>, NdDenseRepr<E::Complex>);
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar + Lapack> RuntimeFor<SolveEigRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = ();

    fn ctx(&self) -> Self::Ctx {}
}

impl<'a, E: Scalar + Lapack> Context<Nd, NormRepr<NdDenseViewRepr<'a, E>>> for () {
    type Output = Result<E::Real, Infallible>;

    fn execute(self, task: NormRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        Ok(task.into_raw().data.norm_l2())
    }
}

impl<'a, E: Scalar + Lapack> RuntimeFor<NormRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = ();

    fn ctx(&self) -> Self::Ctx {}
}

impl<'a, E: Scalar> Context<Nd, ConjRepr<NdDenseViewRepr<'a, E>>> for () {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: ConjRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        Ok(NdDenseRepr {
            data: conj(&task.into_raw().data)?,
        })
    }
}

unsafe impl<'a, E: Scalar> ReprContext<Nd, 1, ConjRepr<NdDenseViewRepr<'a, E>>> for () {
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar> RuntimeFor<ConjRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = ();

    fn ctx(&self) -> Self::Ctx {}
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Diag;

impl<'a, E: Scalar> Context<Nd, ExpRepr<NdDenseViewRepr<'a, E>>> for Diag {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: ExpRepr<NdDenseViewRepr<'a, E>>) -> Self::Output {
        let (a, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.exp())?;
        Ok(NdDenseRepr { data: raw })
    }
}

unsafe impl<'a, E: Scalar> ReprContext<Nd, 1, ExpRepr<NdDenseViewRepr<'a, E>>> for Diag {
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar> RuntimeFor<ExpRepr<NdDenseViewRepr<'a, E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = Diag;

    fn ctx(&self) -> Self::Ctx {
        Diag
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PowF;
//#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
//pub struct DiagPowC;

impl<'a, E: Scalar> Context<Nd, PowRepr<NdDenseViewRepr<'a, E>, E>> for Diag {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: PowRepr<NdDenseViewRepr<'a, E>, E>) -> Self::Output {
        let (a, power, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.pow(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}

unsafe impl<'a, E: Scalar> ReprContext<Nd, 1, PowRepr<NdDenseViewRepr<'a, E>, E>> for Diag {
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar> RuntimeFor<PowRepr<NdDenseViewRepr<'a, E>, E>> for NdRuntime {
    type Mk = Nd;
    type Ctx = Diag;

    fn ctx(&self) -> Self::Ctx {
        Diag
    }
}

impl<'a, E: Scalar> Context<Nd, PowRepr<NdDenseViewRepr<'a, E>, E::Real>> for (Diag, PowF) {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: PowRepr<NdDenseViewRepr<'a, E>, E::Real>) -> Self::Output {
        let (a, power, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.powf(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}

unsafe impl<'a, E: Scalar>
    ReprContext<Nd, 1, PowRepr<NdDenseViewRepr<'a, E>, E::Real>> for (Diag, PowF)
{
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PowI;

impl<'a, E: Scalar> Context<Nd, PowRepr<NdDenseViewRepr<'a, E>, i32>> for (Diag, PowI) {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: PowRepr<NdDenseViewRepr<'a, E>, i32>) -> Self::Output {
        let (a, power, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.powi(power))?;
        Ok(NdDenseRepr { data: raw })
    }
}

unsafe impl<'a, E: Scalar> ReprContext<Nd, 1, PowRepr<NdDenseViewRepr<'a, E>, i32>>
    for (Diag, PowI)
{
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Half;

impl<'a, E: Scalar> Context<Nd, PowRepr<NdDenseViewRepr<'a, E>, Half>> for Diag {
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(self, task: PowRepr<NdDenseViewRepr<'a, E>, Half>) -> Self::Output {
        let (a, _power, axes_split) = task.into_raw();
        let (_, [l_set, r_set]) = axes_split.into_raw();
        let l_set_len = l_set.len();
        let a_idxv_ordered: Vec<usize> = l_set.iter().cloned().chain(r_set.iter().cloned()).collect();
        let a_rot = a.data.permuted_axes(a_idxv_ordered);
        let raw = diag_map(a_rot, l_set_len, |x| *x = x.sqrt())?;
        Ok(NdDenseRepr { data: raw })
    }
}

unsafe impl<'a, E: Scalar> ReprContext<Nd, 1, PowRepr<NdDenseViewRepr<'a, E>, Half>> for Diag {
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'a, E: Scalar> RuntimeFor<PowRepr<NdDenseViewRepr<'a, E>, Half>> for NdRuntime {
    type Mk = Nd;
    type Ctx = Diag;

    fn ctx(&self) -> Self::Ctx {
        Diag
    }
}

#[cfg(test)]
mod tests {

    use std::println;

    use anyhow::Ok;
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
        NdDenseTensor, NdDenseTensorExt,
        linalg::{Diag, Hermite},
    };
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

        let (u, s, v) = t.view().svd(ls![&a, &b], us, vs)?.exec()?.unpack();

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

        let (q, r) = (&t).qr(ls![&a, &b], qr_leg)?.with(())?.unpack();

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
            .with(Hermite)?
            .unpack();

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
            .with(())?
            .unpack();

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
        //let t = Tensor::eye(lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n])?;

        for ai in 0..a_n {
            for bi in 0..b_n {
                t[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]] = (ai + bi) as f64;
            }
        }

        println!("{:?}", t.repr().data.shape());

        let t_exp = (&t)
            .exp(ls![(&a, &a.prime()), (&b, &b.prime())])?
            .with(Diag); // diagonal exp

        {
            let t = Tensor::eye(lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n])?; // [a,a',b,b']
            //let t = Tensor::<f64>::zero(lm![a=>a_n,a.prime()=>a_n,b=>b_n,b.prime()=>b_n])?; // [a,a',b,b']

            println!("{:?}", t.repr());

            let n = (&t)
                .exp(ls![(&a, &a.prime()), (&b, &b.prime())])?
                .with(Diag)?; // scalar
        }

        Ok(())
    }
}
