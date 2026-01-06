use ndarray::{ArrayBase, ArrayView1, ArrayView2, Data, Ix, Ix1, Ix2, s};
use ndarray_linalg::Scalar;
use num_traits::{ConstZero, clamp};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CutFilter<E: Scalar> {
    pub min_ix: Option<Ix>,
    pub max_ix: Option<Ix>,
    pub cutoff: Option<E::Real>,
}
impl<E: Scalar> Default for CutFilter<E> {
    fn default() -> Self {
        CutFilter {
            min_ix: None,
            max_ix: None,
            cutoff: None,
        }
    }
}

use crate::TenalgErr;

use super::Result;

pub trait SingularFilter<SU: Data, SS: Data, SV: Data> {
    fn filter<'u, 's, 'v>(
        self,
        u_mat: &'u ArrayBase<SU, Ix2>,
        s_vec: &'s ArrayBase<SS, Ix1>,
        v_mat: &'v ArrayBase<SV, Ix2>,
    ) -> Result<(
        ArrayView2<'u, SU::Elem>,
        ArrayView1<'s, SS::Elem>,
        ArrayView2<'v, SV::Elem>,
        Ix,
    )>;
}

impl<SU: Data, SS: Data, SV: Data> SingularFilter<SU, SS, SV> for CutFilter<SS::Elem>
where
    SS::Elem: Scalar,
    <SS::Elem as Scalar>::Real: ConstZero,
{
    fn filter<'u, 's, 'v>(
        self,
        u_mat: &'u ArrayBase<SU, Ix2>,
        s_vec: &'s ArrayBase<SS, Ix1>,
        v_mat: &'v ArrayBase<SV, Ix2>,
    ) -> Result<(
        ArrayView2<'u, SU::Elem>,
        ArrayView1<'s, SS::Elem>,
        ArrayView2<'v, SV::Elem>,
        Ix,
    )>
// where
    //     SS::Elem: Scalar,
    //     <SS::Elem as Scalar>::Real: ConstZero,
    {
        // clamp range
        let s_min_ix = self.min_ix.unwrap_or(1);
        let s_max_ix = self.max_ix.unwrap_or(s_vec.shape()[0]);
        if s_min_ix > s_max_ix || s_min_ix < 1 {
            return Err(TenalgErr::InvalidInput);
        }

        let mut s_sqsum = <SS::Elem as Scalar>::Real::ZERO;
        let mut s_valid_ix = 0;

        let mut sq_pre = <SS::Elem as Scalar>::Real::ZERO;
        for i in 0..s_vec.len() {
            let sq = s_vec[i].square();
            if !(sq >= <SS::Elem as Scalar>::Real::ZERO) {
                // negative or nan sing
                return Err(TenalgErr::InvalidResult);
            } else if i != 0 && sq_pre < sq {
                // sing not ordered
                return Err(TenalgErr::InvalidResult);
            }
            if sq > <SS::Elem as Scalar>::Real::ZERO {
                s_valid_ix += 1;
            }
            s_sqsum += sq;
            sq_pre = sq;
        }
        let s_sqsum = s_sqsum;
        // if s_sqsum==0 then s_valid_ix==0, so we can ignore this situation

        // j: cut out small sings
        let mut s_major_ix = s_valid_ix;
        if let Some(s_cutoff) = self.cutoff {
            let mut s_sqsum_err = <SS::Elem as Scalar>::Real::ZERO;
            for i in (0..s_valid_ix).rev() {
                let sq = s_vec[i].square();
                s_sqsum_err += sq;
                if s_sqsum_err / s_sqsum < s_cutoff {
                    s_major_ix = i;
                }
                //println!("i={}, nrsub / nr ={} / {} = {}", i, nrsub, nr, nrsub / nr);
            }
        }
        let s_major_ix = s_major_ix;
        // println!(
        //     "s_dim <- {} {} {} {}",
        //     s_min_dim, s_max_dim, ns_active, ns_large
        // );
        let s_ix = clamp(s_major_ix, s_min_ix, s_max_ix);

        let u_mat_narrowed = u_mat.slice(s![.., 0..s_ix]);
        let s_vec_narrowed = s_vec.slice(s![0..s_ix]);
        let v_mat_narrowed = v_mat.slice(s![0..s_ix, ..]);

        Ok((u_mat_narrowed, s_vec_narrowed, v_mat_narrowed, s_ix))
    }
}

pub fn max_ix<E: Scalar>(max_ix: Ix) -> CutFilter<E> {
    CutFilter {
        min_ix: None,
        max_ix: Some(max_ix),
        cutoff: None,
    }
}
pub fn min_ix<E: Scalar>(min_ix: Ix) -> CutFilter<E> {
    CutFilter {
        min_ix: Some(min_ix),
        max_ix: None,
        cutoff: None,
    }
}
pub fn clamp_ix<E: Scalar>(min_ix: Ix, max_ix: Ix) -> CutFilter<E> {
    CutFilter {
        min_ix: Some(min_ix),
        max_ix: Some(max_ix),
        cutoff: None,
    }
}

pub fn cutoff_sq<E: Scalar>(cutoff: E::Real) -> CutFilter<E> {
    CutFilter {
        min_ix: None,
        max_ix: None,
        cutoff: Some(cutoff),
    }
}

impl<SU: Data, SS: Data, SV: Data> SingularFilter<SU, SS, SV> for ()
where
    SS::Elem: Scalar,
    <SS::Elem as Scalar>::Real: ConstZero,
{
    fn filter<'u, 's, 'v>(
        self,
        u_mat: &'u ArrayBase<SU, Ix2>,
        s_vec: &'s ArrayBase<SS, Ix1>,
        v_mat: &'v ArrayBase<SV, Ix2>,
    ) -> Result<(
        ArrayView2<'u, SU::Elem>,
        ArrayView1<'s, SS::Elem>,
        ArrayView2<'v, SV::Elem>,
        Ix,
    )> {
        (CutFilter::default()).filter(u_mat, s_vec, v_mat)
    }
}
