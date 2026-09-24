use alloc::vec::Vec;
use core::{
    convert::Infallible,
    ops::{Add, Div, Mul, Neg, Sub},
};

use ndarray::{ScalarOperand, Zip};
use ndarray_linalg::{Lapack, Scalar};
use num_traits::ConstZero;
use tensory_core::{
    arith::{
        AddRepr, LeftScalarDivOp, LeftScalarDivRepr, LeftScalarMulOp, LeftScalarMulRepr, MulRepr,
        NegRepr, RightScalarDivOp, RightScalarDivRepr, RightScalarMulOp, RightScalarMulRepr,
        SubRepr,
    },
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::{Context, RuntimeFor},
    },
    repr::ReprContext,
};

use crate::{
    Nd, NdDenseRepr, NdDenseViewRepr, NdRuntime,
    tenalg::{error::TenalgErr, mul},
};

impl<'l, 'r, E> Context<Nd, AddRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for ()
where
    &'l E: Add<&'r E, Output = E>,
{
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(
        self,
        task: AddRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_mapping], _) = task.into_raw();

        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

        let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
        let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

        if lhs_raw.dim() == rhs_raw.dim() {
            Ok(NdDenseRepr {
                data: Zip::from(lhs_raw).and(rhs_raw).map_collect(|l, r| l + r),
            })
        } else {
            Err(TenalgErr::InvalidInput)
        }
    }
}

unsafe impl<'l, 'r, E>
    ReprContext<Nd, 1, AddRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for ()
where
    &'l E: Add<&'r E, Output = E>,
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, TenalgErr>;
}

impl<'l, 'r, E> Context<Nd, SubRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for ()
where
    &'l E: Sub<&'r E, Output = E>,
{
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(
        self,
        task: SubRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_mapping], _) = task.into_raw();

        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

        let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
        let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

        if lhs_raw.dim() == rhs_raw.dim() {
            Ok(NdDenseRepr {
                data: Zip::from(lhs_raw).and(rhs_raw).map_collect(|l, r| l - r),
            })
        } else {
            Err(TenalgErr::InvalidInput)
        }
    }
}

unsafe impl<'l, 'r, E>
    ReprContext<Nd, 1, SubRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for ()
where
    &'l E: Sub<&'r E, Output = E>,
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, TenalgErr>;
}

impl<E: Neg<Output = E> + Clone> Context<Nd, NegRepr<1, NdDenseRepr<E>>> for () {
    type Output = Result<NdDenseRepr<E>, Infallible>;

    fn execute(self, task: NegRepr<1, NdDenseRepr<E>>) -> Self::Output {
        let (a, [axis_alloc], _) = task.into_raw();
        let (_, [perm]) = axis_alloc.into_raw();

        let ndr = -(a.data);

        let ndr = ndr.permuted_axes(perm);
        Ok(NdDenseRepr { data: ndr })
    }
}

unsafe impl<E: Neg<Output = E> + Clone> ReprContext<Nd, 1, NegRepr<1, NdDenseRepr<E>>> for () {
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, Infallible>;
}

impl<E: ScalarOperand + Mul<Output = E> + Clone>
    Context<Nd, LeftScalarMulRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Output = Result<NdDenseRepr<E>, Infallible>;

    fn execute(self, task: LeftScalarMulRepr<1, NdDenseRepr<E>, E>) -> Self::Output {
        let (a, [axis_alloc], LeftScalarMulOp(scalar)) = task.into_raw();
        let (_, [perm]) = axis_alloc.into_raw();

        let ndr = a.data * scalar;
        let ndr = ndr.permuted_axes(perm);
        Ok(NdDenseRepr { data: ndr })
    }
}

unsafe impl<E: ScalarOperand + Mul<Output = E> + Clone>
    ReprContext<Nd, 1, LeftScalarMulRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, Infallible>;
}

impl<E: ScalarOperand + Mul<Output = E> + Clone>
    Context<Nd, RightScalarMulRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Output = Result<NdDenseRepr<E>, Infallible>;

    fn execute(self, task: RightScalarMulRepr<1, NdDenseRepr<E>, E>) -> Self::Output {
        let (a, [axis_alloc], RightScalarMulOp(scalar)) = task.into_raw();
        let (_, [perm]) = axis_alloc.into_raw();

        let ndr = a.data * scalar;
        let ndr = ndr.permuted_axes(perm);
        Ok(NdDenseRepr { data: ndr })
    }
}

unsafe impl<E: ScalarOperand + Mul<Output = E> + Clone>
    ReprContext<Nd, 1, RightScalarMulRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, Infallible>;
}

impl<E: ScalarOperand + Div<Output = E> + Clone>
    Context<Nd, LeftScalarDivRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Output = Result<NdDenseRepr<E>, Infallible>;

    fn execute(self, task: LeftScalarDivRepr<1, NdDenseRepr<E>, E>) -> Self::Output {
        let (a, [axis_alloc], LeftScalarDivOp(scalar)) = task.into_raw();
        let (_, [perm]) = axis_alloc.into_raw();

        let ndr = a.data / scalar;
        let ndr = ndr.permuted_axes(perm);
        Ok(NdDenseRepr { data: ndr })
    }
}

unsafe impl<E: ScalarOperand + Div<Output = E> + Clone>
    ReprContext<Nd, 1, LeftScalarDivRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, Infallible>;
}

impl<E: ScalarOperand + Div<Output = E> + Clone>
    Context<Nd, RightScalarDivRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Output = Result<NdDenseRepr<E>, Infallible>;

    fn execute(self, task: RightScalarDivRepr<1, NdDenseRepr<E>, E>) -> Self::Output {
        let (a, [axis_alloc], RightScalarDivOp(scalar)) = task.into_raw();
        let (_, [perm]) = axis_alloc.into_raw();

        let ndr = a.data / scalar;
        let ndr = ndr.permuted_axes(perm);
        Ok(NdDenseRepr { data: ndr })
    }
}

unsafe impl<E: ScalarOperand + Div<Output = E> + Clone>
    ReprContext<Nd, 1, RightScalarDivRepr<1, NdDenseRepr<E>, E>> for ()
{
    type Repr = NdDenseRepr<E>;

    type CType = Resulting<Raw, Infallible>;
}

impl<'l, 'r, E: Scalar> Context<Nd, MulRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>>
    for ()
{
    type Output = Result<NdDenseRepr<E>, TenalgErr>;

    fn execute(
        self,
        task: MulRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_origin], _) = task.into_raw();

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
                return Err(TenalgErr::InvalidInput);
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

        // #[cfg(test)]
        // {
        //     use std::println;
        //     println!("{:?}", lhs_rot.dim());
        //     println!("{:?}", lhs_idxv_ordered);
        //     println!("{:?}", rhs_rot.dim());
        //     println!("{:?}", rhs_idxv_ordered);

        //     //println!("{:?}", lhs_rem_idxa);
        //     //println!("{:?}", rhs_rem_idxa);
        // }

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

unsafe impl<'l, 'r, E: Scalar>
    ReprContext<Nd, 1, MulRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for ()
{
    type Repr = NdDenseRepr<E>;
    type CType = Resulting<Raw, TenalgErr>;
}

impl<'l, 'r, E> RuntimeFor<AddRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for NdRuntime
where
    &'l E: Add<&'r E, Output = E>,
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}
impl<'l, 'r, E> RuntimeFor<SubRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for NdRuntime
where
    &'l E: Sub<&'r E, Output = E>,
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}
impl<'l, 'r, E: Scalar + Lapack + ConstZero>
    RuntimeFor<MulRepr<1, NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>>> for NdRuntime
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}
impl<E: Neg<Output = E> + Clone> RuntimeFor<NegRepr<1, NdDenseRepr<E>>> for NdRuntime {
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}

impl<E: ScalarOperand + Mul<Output = E> + Clone> RuntimeFor<LeftScalarMulRepr<1, NdDenseRepr<E>, E>>
    for NdRuntime
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}
impl<E: ScalarOperand + Mul<Output = E> + Clone>
    RuntimeFor<RightScalarMulRepr<1, NdDenseRepr<E>, E>> for NdRuntime
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}

impl<E: ScalarOperand + Div<Output = E> + Clone> RuntimeFor<LeftScalarDivRepr<1, NdDenseRepr<E>, E>>
    for NdRuntime
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}
impl<E: ScalarOperand + Div<Output = E> + Clone>
    RuntimeFor<RightScalarDivRepr<1, NdDenseRepr<E>, E>> for NdRuntime
{
    type Mk = Nd;
    type Ctx = ();
    fn ctx(&self) -> Self::Ctx {}
}

#[cfg(test)]
mod tests {
    use std::println;

    use num_traits::abs;
    use tensory_core::prelude::*;

    use crate::{Nd, NdDenseTensor, NdDenseTensorExt, NdRuntime};

    use tensory_basic::{
        id::{Id128, Prime},
        mapper::VecMapper,
    };

    const EPS: f64 = 1e-8;

    type Leg = Prime<Id128>;

    type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;

    #[test]
    fn tensor_add_test() -> Result<(), anyhow::Error> {
        let a_n = 3;
        let b_n = 4;
        let c_n = 5;
        let d_n = 6;

        let a = Leg::new();
        let b = Leg::new();
        let c = Leg::new();
        let d = Leg::new();

        let ta = Tensor::random(lm![a => a_n, b => b_n, c => c_n, d => d_n]).unwrap();
        let tb = Tensor::random(lm![b => b_n, c => c_n, d => d_n, a => a_n]).unwrap();

        let alv = ta.mapper();
        let blv = tb.mapper();
        println!("{:?}", alv);
        println!("{:?}", blv);

        //println!("{:?}", tc.leg_alloc());

        let tc = (&ta + &tb)?.exec()?;

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
                        let tc_e = ta.get(lm![&a=>ai, &b=>bi, &c=>ci, &d=>di])??
                            + tb.get(lm![&a=>ai, &b=>bi, &c=>ci, &d=>di])??;
                        let tc_r = tc.get(lm![&a=>ai, &b=>bi, &c=>ci, &d=>di])??;
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

        let ta = Tensor::random(lm![a => a_n, i => i_n, b => b_n, j => j_n]).unwrap();
        let tb = Tensor::random(lm![j => j_n, c => c_n, d => d_n, i => i_n]).unwrap();

        let nd = NdRuntime;

        let alv = ta.mapper();
        let blv = tb.mapper();
        println!("{:?}", alv);
        println!("{:?}", blv);

        let ta = ta.bind(nd);
        let tb = tb.bind(nd);

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
                                tc_e += ta.get(lm![&a=>ai, &i=>ii, &b=>bi, &j=>ji])??
                                    * tb.get(lm![&j=>ji, &c=>ci, &d=>di, &i=>ii])??;
                            }
                        }
                        let tc_r = tc.get(lm![&a=>ai, &b=>bi, &c=>ci, &d=>di])??;
                        //tcd[[ai, bi, ci, di]];
                        println!("{},{},{},{} : {} vs {}", ai, bi, ci, di, tc_e, tc_r);
                        assert!(abs(tc_e - tc_r) < EPS);
                    }
                }
            }
        }

        Ok(())
    }
}
