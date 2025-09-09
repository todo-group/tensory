mod convert;
pub mod cut_filter;
pub mod error;

use alloc::vec::Vec;
use convert::{mat_to_ten, ten_to_mat};
use cut_filter::CutFilter;
use error::TenalgError;
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayD, ArrayView1, ArrayView2, CowArray, Data, Dimension,
    ErrorKind::IncompatibleShape, Ix, Ix1, Ix2, ShapeError, linalg::Dot, s,
};
use ndarray_linalg::{QR, SVDInto, Scalar, conjugate, svd::SVD};
use num_traits::{ConstZero, clamp};

type Result<T> = core::result::Result<T, TenalgError>;

pub fn conj<S: Data, D: Dimension>(x: &ArrayBase<S, D>) -> Result<Array<S::Elem, D>>
where
    S::Elem: Clone + Scalar,
{
    let x_full_ix = x.shape().into_iter().product();

    let x_mat = ten_to_mat(x, [x_full_ix, 1])?;
    let x_conj_mat: Array2<S::Elem> = conjugate(&x_mat);
    let x_conj: Array<S::Elem, D> = mat_to_ten(&x_conj_mat, x.raw_dim())?.into_owned();
    Ok(x_conj)
}

pub fn mul<S1: Data, D1: Dimension, S2: Data, D2: Dimension, SM: Data>(
    x: &ArrayBase<S1, D1>,
    y: &ArrayBase<S2, D2>,
    concat_dim: usize,
) -> Result<ArrayD<SM::Elem>>
where
    S1::Elem: Clone,
    S2::Elem: Clone,
    SM::Elem: Clone,
    for<'x, 'y> CowArray<'x, S1::Elem, Ix2>:
        Dot<CowArray<'y, S2::Elem, Ix2>, Output = ArrayBase<SM, Ix2>>,
{
    let x_ixs = x.shape();
    let y_ixs = y.shape();
    let x_dim = x_ixs.len();
    let y_dim = y_ixs.len();

    if x_dim < concat_dim || y_dim < concat_dim {
        return Err(ShapeError::from_kind(IncompatibleShape).into());
    }
    let (x_remain_ixs, x_concat_ixs) = x_ixs.split_at(x_dim - concat_dim);
    let (y_concat_ixs, y_remain_ixs) = y_ixs.split_at(concat_dim);

    if x_concat_ixs != y_concat_ixs {
        return Err(ShapeError::from_kind(IncompatibleShape).into());
    }
    let z_ixs = [x_remain_ixs, y_remain_ixs].concat();

    let x_remain_full_ix: Ix = x_remain_ixs.iter().product();
    let y_remain_full_ix: Ix = y_remain_ixs.iter().product();
    let concat_full_ix: Ix = x_concat_ixs.iter().product();

    let x_mat = ten_to_mat(x, [x_remain_full_ix, concat_full_ix])?;
    let y_mat = ten_to_mat(y, [concat_full_ix, y_remain_full_ix])?;

    let z_mat = x_mat.dot(&y_mat);
    let z = mat_to_ten(&z_mat, z_ixs)?.into_owned();
    Ok(z)
}

fn filter_singular_value<'u, 's, 'v, SU: Data, SS: Data, SV: Data, F: CutFilter<SS::Elem>>(
    u_mat: &'u ArrayBase<SU, Ix2>,
    s_vec: &'s ArrayBase<SS, Ix1>,
    v_mat: &'v ArrayBase<SV, Ix2>,
    s_filter: F,
) -> Result<(
    ArrayView2<'u, SU::Elem>,
    ArrayView1<'s, SS::Elem>,
    ArrayView2<'v, SV::Elem>,
    Ix,
)>
where
    SS::Elem: Scalar,
    <SS::Elem as Scalar>::Real: ConstZero,
{
    // clamp range
    let s_min_ix = s_filter.min_ix().unwrap_or(1);
    let s_max_ix = s_filter.max_ix().unwrap_or(s_vec.shape()[0]);
    if s_min_ix > s_max_ix || s_min_ix < 1 {
        return Err(TenalgError::InvalidInput);
    }

    let mut s_sqsum = <SS::Elem as Scalar>::Real::ZERO;
    let mut s_valid_ix = 0;

    let mut sq_pre = <SS::Elem as Scalar>::Real::ZERO;
    for i in 0..s_vec.len() {
        let sq = s_vec[i].square();
        if !(sq >= <SS::Elem as Scalar>::Real::ZERO) {
            // negative or nan sing
            return Err(TenalgError::InvalidResult);
        } else if i != 0 && sq_pre < sq {
            // sing not ordered
            return Err(TenalgError::InvalidResult);
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
    if let Some(s_cutoff) = s_filter.cutoff() {
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

pub fn svd<S: Data, D: Dimension, SU: Data, SS: Data, SV: Data, F: CutFilter<SS::Elem>>(
    x: &ArrayBase<S, D>,
    left_dim: usize,
    s_filter: F,
) -> Result<(ArrayD<SU::Elem>, Array1<SS::Elem>, ArrayD<SV::Elem>)>
where
    S::Elem: Clone,
    SU::Elem: Clone,
    SV::Elem: Clone,
    SS::Elem: Clone + Scalar,
    <SS::Elem as Scalar>::Real: ConstZero,
    for<'x> CowArray<'x, S::Elem, Ix2>:
        SVD<U = ArrayBase<SU, Ix2>, Sigma = ArrayBase<SS, Ix1>, VT = ArrayBase<SV, Ix2>>,
{
    let x_ixs = x.shape();
    let x_dim = x_ixs.len();

    if x_dim < left_dim {
        return Err(ShapeError::from_kind(IncompatibleShape).into());
    }
    let (left_ixs, right_ixs) = x_ixs.split_at(left_dim);

    let left_full_ix: Ix = left_ixs.iter().product();
    let right_full_ix: Ix = right_ixs.iter().product();

    let x_mat = ten_to_mat(x, [left_full_ix, right_full_ix])?;
    if let (Some(u_mat), s_vec, Some(v_mat)) = x_mat.svd(true, true)? {
        let (u_mat_narrowed, s_vec_narrowed, v_mat_narrowed, s_ix) =
            filter_singular_value(&u_mat, &s_vec, &v_mat, s_filter)?;

        let u_ixs: Vec<Ix> = [left_ixs, &[s_ix]].concat();
        let v_ixs: Vec<Ix> = [&[s_ix], right_ixs].concat();

        let u = mat_to_ten(&u_mat_narrowed, u_ixs)?.into_owned();
        let s = s_vec_narrowed.into_owned();
        let v = mat_to_ten(&v_mat_narrowed, v_ixs)?.into_owned();
        Ok((u, s, v))
    } else {
        Err(TenalgError::InvalidResult)
    }
}

pub fn into_svd<S: Data, D: Dimension, SU: Data, SS: Data, SV: Data, F: CutFilter<SS::Elem>>(
    x: ArrayBase<S, D>,
    left_dim: usize,
    s_filter: F,
) -> Result<(ArrayD<SU::Elem>, Array1<SS::Elem>, ArrayD<SV::Elem>)>
where
    S::Elem: Clone,
    SU::Elem: Clone,
    SV::Elem: Clone,
    SS::Elem: Clone + Scalar,
    <SS::Elem as Scalar>::Real: ConstZero,
    for<'x> CowArray<'x, S::Elem, Ix2>:
        SVDInto<U = ArrayBase<SU, Ix2>, Sigma = ArrayBase<SS, Ix1>, VT = ArrayBase<SV, Ix2>>,
{
    let x_ixs = x.shape();
    let x_dim = x_ixs.len();

    if x_dim < left_dim {
        return Err(ShapeError::from_kind(IncompatibleShape).into());
    }
    let (left_ixs, right_ixs) = x_ixs.split_at(left_dim);

    let left_full_ix: Ix = left_ixs.iter().product();
    let right_full_ix: Ix = right_ixs.iter().product();

    let x_mat = ten_to_mat(&x, [left_full_ix, right_full_ix])?;
    if let (Some(u_mat), s_vec, Some(v_mat)) = x_mat.svd_into(true, true)? {
        let (u_mat_narrowed, s_vec_narrowed, v_mat_narrowed, s_ix) =
            filter_singular_value(&u_mat, &s_vec, &v_mat, s_filter)?;

        let u_ixs: Vec<Ix> = [left_ixs, &[s_ix]].concat();
        let v_ixs: Vec<Ix> = [&[s_ix], right_ixs].concat();

        let u = mat_to_ten(&u_mat_narrowed, u_ixs)?.into_owned();
        let s = s_vec_narrowed.into_owned();
        let v = mat_to_ten(&v_mat_narrowed, v_ixs)?.into_owned();
        Ok((u, s, v))
    } else {
        Err(TenalgError::InvalidResult)
    }
}

pub fn qr<S: Data, D: Dimension, SQ: Data, SR: Data>(
    x: &ArrayBase<S, D>,
    left_dim: usize,
) -> Result<(ArrayD<SQ::Elem>, ArrayD<SR::Elem>)>
where
    S::Elem: Clone,
    SQ::Elem: Clone,
    SR::Elem: Clone,
    for<'x> CowArray<'x, S::Elem, Ix2>: QR<Q = ArrayBase<SQ, Ix2>, R = ArrayBase<SR, Ix2>>,
{
    let x_ixs = x.shape();

    if x_ixs.len() < left_dim {
        return Err(ShapeError::from_kind(IncompatibleShape).into());
    }

    let (left_ixs, right_ixs) = x_ixs.split_at(left_dim);

    let left_full_ix: usize = left_ixs.iter().product();
    let right_full_ix: usize = right_ixs.iter().product();

    let x_mat = ten_to_mat(&x, [left_full_ix, right_full_ix])?;

    let (q_mat, r_mat) = x_mat.qr()?;

    let inner_ix = r_mat.shape()[0];

    let q_ixs: Vec<Ix> = [left_ixs, &[inner_ix]].concat();
    let r_ixs: Vec<Ix> = [&[inner_ix], right_ixs].concat();

    let q = mat_to_ten(&q_mat, q_ixs)?.into_owned();
    let r = mat_to_ten(&r_mat, r_ixs)?.into_owned();
    Ok((q, r))
}

#[cfg(test)]
mod tests {

    use super::*;
    use cut_filter::{Cutoff, MaxIx};
    //use linalg_ext::SVD;
    use ndarray::{Array, ArrayD, Ix5, IxDyn};
    use ndarray_linalg::{Norm, from_diag, random};
    use num_complex::Complex;
    use std::boxed::Box;
    use std::println;
    use std::{error, f64::NAN};
    type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

    #[test]
    fn zero_test() {
        let ze: f64 = 0.0;
        let zc: f64 = f64::ZERO;
        let nz: f64 = -0.0;

        let na: f64 = NAN;

        assert!((ze == ze));
        assert!((ze == zc));
        assert!((zc == ze));
        assert!((zc == zc));

        assert!((nz == nz));
        assert!((ze == nz));
        assert!((nz == ze));
        assert!((zc == nz));
        assert!((nz == zc));

        assert!((na != ze));
        assert!((na != zc));
        assert!((na != nz));
        assert!((ze != na));
        assert!((zc != na));
        assert!((nz != na));
        assert!((na != na));

        assert!(!(ze < ze));
        assert!(!(ze < zc));
        assert!(!(zc < ze));
        assert!(!(zc < zc));

        assert!(!(nz < nz));
        assert!(!(ze < nz));
        assert!(!(nz < ze));
        assert!(!(zc < nz));
        assert!(!(nz < zc));

        assert!(!(na < ze));
        assert!(!(na < zc));
        assert!(!(na < nz));
        assert!(!(ze < na));
        assert!(!(zc < na));
        assert!(!(nz < na));
        assert!(!(na < na));
    }

    #[test]
    fn conj_test() -> Result<()> {
        let x: Array<Complex<f64>, Ix5> = random([2, 3, 4, 5, 6]);
        println!("{:?}", x.shape());
        {
            let x_conj = conj(&x)?;
            for i in 0..2 {
                for j in 0..3 {
                    for k in 0..4 {
                        for l in 0..5 {
                            for m in 0..6 {
                                assert!(x[[i, j, k, l, m]].conj() == x_conj[[i, j, k, l, m]]);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[test]
    fn mul_test() -> Result<()> {
        let mut x = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4]));
        let mut x_orig = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4]));
        //let mut y = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3]));
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    x[[i, j, k]] = (4 * i + 2 * j + k) as f64;
                    x_orig[[i, j, k]] = (4 * i + 2 * j + k) as f64;
                }
            }
        }
        let mut y = ArrayD::<f64>::zeros(IxDyn(&[6, 5, 4]));
        let mut y_orig = ArrayD::<f64>::zeros(IxDyn(&[6, 5, 4]));
        for i in 0..6 {
            for j in 0..5 {
                for k in 0..4 {
                    y[[i, j, k]] = (4 * i + 2 * j + k) as f64;
                    y_orig[[i, j, k]] = (4 * i + 2 * j + k) as f64;
                    //print!("{} ", y[[i, j, k]]);
                }
                //println!("");
            }
            //println!("");
        }
        //println!("");

        println!("{:?} * {:?}", x.shape(), y.shape());

        //let y_orig = y_orig.clone_from(y);
        y.swap_axes(0, 2);
        y.swap_axes(1, 2);
        x.swap_axes(0, 1);

        println!("-> {:?} * {:?}", x.shape(), y.shape());

        //println!("");
        //let y_dum = y_orig.into_shape([4, 5, 6])?;

        //let z = tenmul(x.clone(), y.clone(), 1)?;

        // let x_mat = x
        //     .to_shape(([2 * 3, 4], Order::ColumnMajor))
        //     .and_then(|m| m.into_dimensionality::<Ix2>())?;

        // println!("{}", x_mat);

        // let y_mat = y
        //     .to_shape(([4, 5 * 6], Order::ColumnMajor))
        //     .and_then(|m| m.into_dimensionality::<Ix2>())?;

        // println!("{}", y_mat);

        // for k in 0..4 {
        //     for i in 0..6 {
        //         for j in 0..5 {
        //             //print!("{} ", y[[i, j, k]]);
        //             assert_eq!(y_mat[[k, j + i * 5]], y_orig[[i, j, k]]);
        //         }
        //         //println!("");
        //     }
        //     //println!("");
        // }

        //let z_mat: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
        //     x_mat.dot(&y_mat);
        // println!("{}", z_mat);
        // println!("z_mat: {:?}", z_mat.shape());
        // let z_mat_orig = z_mat.clone();
        // let z = z_mat.to_shape(([2, 3, 5, 6], Order::ColumnMajor))?;

        // for i in 0..2 {
        //     for j in 0..3 {
        //         for jj in 0..5 {
        //             for ii in 0..6 {
        //                 //print!("{} ", y[[i, j, k]]);
        //                 println!(
        //                     "z {} {} {} {} : {} vs {}",
        //                     i,
        //                     j,
        //                     jj,
        //                     ii,
        //                     z_mat_orig[[i + 2 * j, jj + 5 * ii]],
        //                     z[[i, j, jj, ii]]
        //                 );
        //                 assert_eq!(z_mat_orig[[i + 2 * j, jj + 5 * ii]], z[[i, j, jj, ii]]);
        //             }
        //         }
        //         //println!("");
        //     }
        //     //println!("");
        // }

        // //let y_mat =

        let z = mul(&x, &y, 1)?;

        for i in 0..2 {
            for j in 0..3 {
                for ii in 0..6 {
                    for jj in 0..5 {
                        let mut z_expect: f64 = 0.;
                        for k in 0..4 {
                            z_expect += x_orig[[i, j, k]] * y_orig[[ii, jj, k]] as f64;
                        }
                        let z_tenmul = z[[j, i, ii, jj]];
                        println!("{} {} {} {} : {} vs {}", i, j, ii, jj, z_tenmul, z_expect,);
                        assert_eq!(z_tenmul, z_expect);
                    }
                }
            }
        }

        // let a = array![[1., 2.,], [0., 1.,]];
        // let b = array![[1., 2.,], [0., 1.,]];
        // let c = a.dot(&b);
        // assert!(c == array![[5., 8.], [2., 3.]]);
        Ok(())
    }

    #[test]
    fn svd_test() -> Result<()> {
        // let mut x = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4, 5, 6]));
        // //let mut x_orig = ArrayD::<f64>::zeros(IxDyn(&[2, 3, 4]));
        // //let mut y = ArrayD::<f64>::zeros(IxDyn(&[5, 6, 3]));
        // for i in 0..2 {
        //     for j in 0..3 {
        //         for k in 0..4 {
        //             for l in 0..5 {
        //                 for m in 0..5 {
        //                     x[[i, j, k, l, m]] = (4 * i + 2 * j + k) as f64;
        //                     //x_orig[[i, j, k]] = (4 * i + 2 * j + k) as f64;
        //                 }
        //             }
        //         }
        //     }
        // }
        let x: Array<f64, Ix5> = random([2, 3, 4, 5, 6]);
        println!("{:?}", x.shape());

        for i in 1..24 + 1 {
            let (u, s, v) = svd(&x, 3, MaxIx(i))?;
            //println!("u: {:?}", u.shape());
            //println!("s: {:?}", s.shape());
            //println!("v: {:?}", v.shape());
            let ss = from_diag(&s.to_vec());

            //println!("ss: {:?}", ss.shape());
            let us = mul(&u, &ss, 1)?;
            //println!("us: {:?}", us.shape());
            let usv = mul(&us, &v, 1)?;
            //println!("usv: {:?}", usv.shape());
            let orino = x.norm();
            let nodiff = x.norm() - usv.norm();

            let diff = x.clone() - usv;
            println!(
                "[{}] diff: {} or {}/{} = {}",
                i,
                diff.norm(),
                nodiff,
                orino,
                nodiff / orino
            );

            let s_dim = s.shape()[0];
            let mut ut = conj(&u)?;
            //println!("{:?}", u.shape());
            ut.swap_axes(3, 2);
            ut.swap_axes(2, 1);
            ut.swap_axes(1, 0);
            //println!("{:?}", ut.shape());
            let mut uut = mul(&ut, &u, 3)?;
            //println!("{}", uut);
            let eps = 1e-8;
            assert!((i as f64).sqrt() - eps < uut.norm() && uut.norm() < (i as f64).sqrt() + eps);
            for si in 0..s_dim {
                uut[[si, si]] -= 1.0;
            }
            //println!("{}", uut.norm());
            assert!(0.0 <= uut.norm() && uut.norm() < eps);
        }

        {
            let cutoff = 1e-2;
            let (u, s, v) = svd(&x, 3, Cutoff(cutoff))?;
            //println!("u: {:?}", u.shape());
            //println!("s: {:?}", s.shape());
            //println!("v: {:?}", v.shape());
            let ss = from_diag(&s.to_vec());

            //println!("ss: {:?}", ss.shape());
            let us = mul(&u, &ss, 1)?;
            //println!("us: {:?}", us.shape());
            let usv = mul(&us, &v, 1)?;
            //println!("usv: {:?}", usv.shape());
            let orino = x.norm();
            let nodiff = x.norm() - usv.norm();

            let diff = x.clone() - usv;
            println!(
                "[cutoff={}] diff: {} or {}/{} = {} dim={}",
                cutoff,
                diff.norm(),
                nodiff,
                orino,
                nodiff / orino,
                s.shape()[0]
            );
        }

        {
            let (u, s, v) = svd(&x, 3, ())?;
            //println!("u: {:?}", u.shape());
            //println!("s: {:?}", s.shape());
            //println!("v: {:?}", v.shape());
            let ss = from_diag(&s.to_vec());

            //println!("ss: {:?}", ss.shape());
            let us = mul(&u, &ss, 1)?;
            //println!("us: {:?}", us.shape());
            let usv = mul(&us, &v, 1)?;
            //println!("usv: {:?}", usv.shape());
            let orino = x.norm();
            let nodiff = x.norm() - usv.norm();

            let diff = x.clone() - usv;
            println!(
                "[full] diff: {} or {}/{} = {} dim={}",
                diff.norm(),
                nodiff,
                orino,
                nodiff / orino,
                s.shape()[0]
            );
        }

        //let z = tenmul(&x, &y, 1)?;

        // s.iter().for_each(|x| {
        //     println!("{}", x);
        //     assert!(*x != 0.)
        // });

        // for i in 0..2 {
        //     for j in 0..3 {
        //         for k in 0..4 {
        //             for l in 0..5 {
        //                 for m in 0..5 {
        //                     //assert_eq!(x[[i, j, k, l, m]], usv[[i, j, k, l, m]]);
        //                     //x_orig[[i, j, k]] = (4 * i + 2 * j + k) as f64;
        //                 }
        //             }
        //         }
        //     }
        // }

        // for i in 0..2 {
        //     for j in 0..3 {
        //         for ii in 0..6 {
        //             for jj in 0..5 {
        //                 let mut z_expect: f64 = 0.;
        //                 for k in 0..4 {
        //                     z_expect += x_orig[[i, j, k]] * y_orig[[ii, jj, k]] as f64;
        //                 }
        //                 let z_tenmul = z[[j, i, ii, jj]];
        //                 println!("{} {} {} {} : {} vs {}", i, j, ii, jj, z_tenmul, z_expect,);
        //                 assert_eq!(z_tenmul, z_expect);
        //             }
        //         }
        //     }
        // }

        // let a = array![[1., 2.,], [0., 1.,]];
        // let b = array![[1., 2.,], [0., 1.,]];
        // let c = a.dot(&b);

        //let a: Array2<f64> = ndarray_linalg::random((10, 10));
        //let (u, s, vt) = a.svd(true, true).unwrap();

        // assert!(c == array![[5., 8.], [2., 3.]]);
        Ok(())
    }

    #[test]
    fn qr_test() -> Result<()> {
        let x: Array<f64, Ix5> = random([2, 3, 4, 5, 6]);
        println!("{:?}", x.shape());
        {
            let (q, r) = qr(&x, 3)?;
            println!("q: {:?}", q.shape());
            println!("r: {:?}", r.shape());
            let x_again = mul(&q, &r, 1)?;

            let orino = x.norm();
            let nodiff = x.norm() - x_again.norm();

            let diff = x.clone() - x_again;
            println!(
                "[full] diff: {} or {}/{} = {} dim={}",
                diff.norm(),
                nodiff,
                orino,
                nodiff / orino,
                r.shape()[0]
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::println;

    use ndarray::{Array, Ix2};
    use ndarray_linalg::{SVD, random};

    #[test]
    fn svd_speedtest() {
        let x: Array<f64, Ix2> = random([200, 3000]);
        println!("{:?}", x.shape());

        let pre = chrono::Local::now();

        let x = x.svd(true, true).unwrap();

        let post = chrono::Local::now();

        println!("{}", post - pre);

        panic!();
    }
}
