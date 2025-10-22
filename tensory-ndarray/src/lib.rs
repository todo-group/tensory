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
    iter::Flatten,
    ops::{Add, Div, Mul},
    panic,
};

use rand::Rng;
use tensory_core::{
    args::LegMapArg,
    arith::{
        AddCtxImpl, CommutativeScalarDivContext, CommutativeScalarMulContext, ConjCtx, MulRuntime,
    },
    bound_tensor::Runtime,
    mapper::{AxisMapper, BuildableMapper, ConnectAxisOrigin, GroupedAxes},
    repr::{AsViewMutRepr, AsViewRepr},
    tensor::Tensor,
    utils::{
        axis_info::AxisInfoImpl,
        elem_get::{ElemGetMutReprImpl, ElemGetReprImpl},
    },
};

use alloc::vec::Vec;
use ndarray::{ArrayBase, ArrayD, ArrayViewD, ArrayViewMutD, IxDyn, OwnedRepr, ScalarOperand, Zip};
use ndarray_linalg::{
    Lapack, Scalar, from_diag, random, random_hermite, random_hermite_using, random_using,
};
use num_traits::{ConstZero, Zero};
use tensory_core::{arith::MulCtxImpl, repr::TensorRepr};
use tensory_linalg::{eigen::EighContextImpl, qr::QrContextImpl, svd::SvdContextImpl};

use crate::tenalg::{
    conj, cut_filter::CutFilter, error::TenalgError, into_eigh, into_qr, into_svd, into_svddc, mul,
};

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
    fn random_using(sizes: impl Iterator<Item = usize>, rng: &mut impl Rng) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        Self {
            data: random_using(sizes, rng),
        }
    }
    fn random_hermite(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        Self {
            data: random_hermite(full_size)
                .into_shape_with_order(sizes_dup)
                .unwrap(),
        }
    }
    fn random_hermite_using(sizes: impl Iterator<Item = usize>, rng: &mut impl Rng) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        Self {
            data: random_hermite_using(full_size, rng)
                .into_shape_with_order(sizes_dup)
                .unwrap(),
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
    fn map<E2, F: FnMut(&E) -> E2>(&self, f: F) -> NdDenseRepr<E2> {
        NdDenseRepr {
            data: self.data.map(f),
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
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
unsafe impl<E> TensorRepr for NdDenseViewRepr<'_, E> {
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
unsafe impl<E> TensorRepr for NdDenseViewMutRepr<'_, E> {
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
impl<E> AxisInfoImpl for NdDenseRepr<E> {
    type AxisInfo = usize;

    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
        self.data.shape()[i]
    }
}

impl<E> AxisInfoImpl for NdDenseViewRepr<'_, E> {
    type AxisInfo = usize;

    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
        self.data.shape()[i]
    }
}

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

unsafe impl<'l, 'r, E> AddCtxImpl<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>> for ()
where
    for<'a> &'a E: Add<&'a E, Output = E> + Clone,
{
    type Res = NdDenseRepr<E>;
    type Err = TenalgError;

    unsafe fn add_unchecked(
        self,
        lhs: NdDenseViewRepr<'l, E>,
        rhs: NdDenseViewRepr<'r, E>,
        axis_mapping: tensory_core::mapper::OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let lhs_raw = lhs.data;
        let rhs_raw = rhs.data;

        let (_, [lhs_perm, rhs_perm]) = axis_mapping.into_raw();

        let lhs_raw = lhs_raw.permuted_axes(lhs_perm);
        let rhs_raw = rhs_raw.permuted_axes(rhs_perm);

        if lhs_raw.dim() == rhs_raw.dim() {
            Ok(NdDenseRepr {
                data: Zip::from(&lhs_raw).and(&rhs_raw).map_collect(|l, r| l + r),
            })
        } else {
            Err(TenalgError::InvalidInput)
        }
    }
}

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

unsafe impl<'a, E: Scalar + Lapack> QrContextImpl<NdDenseViewRepr<'a, E>> for () {
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

unsafe impl<'a, E: Scalar + Lapack> EighContextImpl<NdDenseViewRepr<'a, E>> for () {
    type VC = NdDenseRepr<E>;
    type D = NdDenseRepr<E::Real>;
    type V = NdDenseRepr<E>;

    type Err = TenalgError;

    unsafe fn eigh_unchecked(
        self,
        a: NdDenseViewRepr<'a, E>,
        axes_split: tensory_core::mapper::EquivGroupedAxes<2>,
    ) -> Result<(Self::VC, Self::D, Self::V), Self::Err> {
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

unsafe impl<'a, E: Scalar> ConjCtx<NdDenseViewRepr<'a, E>> for () {
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

pub trait NdDenseTensorExt<E, M: AxisMapper>: Sized {
    fn zero<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        M: BuildableMapper<K>;
    fn random<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;
    fn random_using<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;

    fn random_hermite<
        K: ExactSizeIterator + Iterator<Item = (M::Id, M::Id)>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;
    fn random_hermite_using<
        K: ExactSizeIterator + Iterator<Item = (M::Id, M::Id)>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;

    fn map<E2, F: FnMut(&E) -> E2>(&self, f: F) -> Tensor<NdDenseRepr<E2>, M>
    where
        M: Clone;
}
impl<E, M: AxisMapper> NdDenseTensorExt<E, M> for NdDenseTensor<E, M> {
    fn zero<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();

        let mapper = M::build(k)?;

        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::zero(v), mapper) })
    }
    fn random<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let broker = M::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random(v), broker) })
    }
    fn random_using<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let broker = M::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random_using(v, rng), broker) })
    }

    fn random_hermite<
        K: ExactSizeIterator + Iterator<Item = (M::Id, M::Id)>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let broker = M::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random_hermite(v), broker) })
    }
    fn random_hermite_using<
        K: ExactSizeIterator + Iterator<Item = (M::Id, M::Id)>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let broker = M::build(k)?;
        Ok(
            unsafe {
                Tensor::from_raw_unchecked(NdDenseRepr::random_hermite_using(v, rng), broker)
            },
        )
    }

    fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> Tensor<NdDenseRepr<E2>, M>
    where
        M: Clone,
    {
        unsafe { Tensor::from_raw_unchecked(self.repr().map(&mut f), self.mapper().clone()) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdRuntime;
unsafe impl Runtime for NdRuntime {}

impl<'l, 'r, E: Scalar + Lapack + ConstZero>
    MulRuntime<NdDenseViewRepr<'l, E>, NdDenseViewRepr<'r, E>> for NdRuntime
{
    type Ctx = ();
    fn mul_ctx(&self) -> Self::Ctx {}
}

#[cfg(test)]
mod tests {
    use std::{println, vec};

    use num_traits::abs;
    use tensory_core::leg;
    use tensory_linalg::{eigen::TensorEighExt, qr::TensorQrExt, svd::TensorSvdExt};

    use super::*;

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

        let ta = Tensor::random(leg![a => a_n, b => b_n, c => c_n, d => d_n]).unwrap();
        let tb = Tensor::random(leg![b => b_n, c => c_n, d => d_n, a => a_n]).unwrap();

        let alv = ta.mapper();
        let blv = tb.mapper();
        println!("{:?}", alv);
        println!("{:?}", blv);

        //println!("{:?}", tc.leg_alloc());

        let tc = (&ta + &tb)?.with(())?;

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
                        let tc_e = ta.get(leg![&a=>ai, &b=>bi, &c=>ci, &d=>di])??
                            + tb.get(leg![&a=>ai, &b=>bi, &c=>ci, &d=>di])??;
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

        let t = Tensor::random(leg![a=>a_n, c=>c_n, d=>d_n, b=>b_n]).unwrap();

        let qr_leg = Leg::new();

        let (q, r) = (&t).qr(leg![&a, &b], qr_leg)?.with(())?;

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
                            (t.get(leg![&a=>ai,&b=>bi,&c=>ci,&d=>di])??
                                - qr.get(leg![&a=>ai,&b=>bi,&c=>ci,&d=>di])??)
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

        let qt = qt.replace_leg(&a, a.prime()).unwrap();
        let qt = qt.replace_leg(&b, b.prime()).unwrap();
        println!("{:?}", qt.mapper());
        let qqt = (&q * qt)?.with(())?;
        println!("{:?}", qqt.mapper());
        for ai in 0..a_n {
            for bi in 0..b_n {
                for api in 0..a_n {
                    for bpi in 0..b_n {
                        let re = *qqt.get(leg![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
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

        let t = Tensor::from_raw(
            NdDenseRepr {
                data: ndarray_linalg::random_hermite(a_n * b_n)
                    .into_shape_with_order(vec![a_n, b_n, a_n, b_n])?,
            },
            VecMapper::build([a, b, a.prime(), b.prime()].into_iter())?,
        )
        .unwrap();

        let vcd = Leg::new();
        let dv = Leg::new();

        let (vc, d, v) = t
            .view()
            .eigh(leg![(&a, &a.prime()), (&b, &b.prime())], vcd, dv)?
            .with(())?;

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
                            (t.get(leg![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi])??
                                - usv
                                    .get(leg![&a=>ai,&b=>bi,&a.prime()=>api,&b.prime()=>bpi])??)
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
        //                 let re = *uut.get(leg![&a=>ai,&b=>bi,&ap=>api,&bp=>bpi])??;
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
}
