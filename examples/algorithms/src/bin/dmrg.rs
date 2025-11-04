use num_complex::Complex;
use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
//use tensory_core::tensor::TensorRepr;
use tensory_core::{args::LegMapArg, leg, tensor::TensorTask};
use tensory_linalg::{conj::TensorConjExt, svd::TensorSvdExt};
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, cut_filter::MaxIx};
type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<Complex<f64>, VecMapper<Leg>>;
fn main() -> anyhow::Result<()> {
    // Example usage of the negation operation
    let n = 10;

    let phys: Vec<_> = (0..n).map(|_| Leg::new()).collect();
    let psi_virt: Vec<_> = (0..n + 1).map(|_| Leg::new()).collect();
    let hamil_virt: Vec<_> = (0..n + 1).map(|_| Leg::new()).collect();

    let mut psi: Vec<_> = (0..n)
        .map(|i| Tensor::random(leg![phys[i]=>2, psi_virt[i]=>1, psi_virt[i + 1]=>1]).unwrap())
        .collect();
    let hamil: Vec<_> = (0..n)
        .map(|i| {
            Tensor::zero(leg![
                phys[i]=>2,
                phys[i].prime()=>2,
                hamil_virt[i]=>1,
                hamil_virt[i + 1]=>1
            ])
            .unwrap()
        })
        .collect();

    let t: usize = 200;

    //let blas: BlasRuntime = todo!(libload);

    for sweep in 0..t {
        std::println!("sweep {}", sweep);
        for i in 0..(n - 1) {
            std::println!("i {}", i);
            let mut left = Tensor::zero(unsafe {
                LegMapArg::from_raw_unchecked([].into_iter(), [].into_iter())
            })
            .unwrap();
            if i > 0 {
                for j in 0..(i - 1) {
                    left = (&left * &psi[j])?.with(())?;
                    left = (&left * &hamil[j])?.with(())?;
                    left = (&left
                        * &(&psi[j])
                            .conj()
                            .with(())?
                            .replace_leg(leg![
                                &phys[j]=> phys[j].prime(),
                                &psi_virt[j]=> psi_virt[j].prime(),
                                &psi_virt[j + 1]=> psi_virt[j + 1].prime()
                            ])
                            .unwrap())?
                        .with(())?;
                }
            }

            let mut right = Tensor::zero(unsafe {
                LegMapArg::from_raw_unchecked([].into_iter(), [].into_iter())
            })
            .unwrap();
            for j in ((i + 2)..n).rev() {
                right = (&right * &psi[j])?.with(())?;
                right = (&right * &hamil[j])?.with(())?;
                right = (&right
                    * &(&psi[j])
                        .conj()
                        .with(())?
                        .replace_leg(leg![
                            &phys[j]=> phys[j].prime(),
                            &psi_virt[j]=> psi_virt[j].prime(),
                            &psi_virt[j + 1]=> psi_virt[j + 1].prime()
                        ])
                        .unwrap())?
                    .with(())?;
            }

            //println!("left {:?}", left.repr().dim());
            //println!("right {:?}", right.repr().dim());

            let center = (&psi[i] * &psi[i + 1])?.with(())?;

            // optimizer using left,right,center,hamil[i],hamil[i+1]

            let dum = Leg::new();
            let (u, s, v) = (&center)
                .svd(leg![&phys[i], &psi_virt[i]], psi_virt[i + 1], dum)?
                .with((MaxIx(5),))?;

            let s = s.map(|e| e.into());

            psi[i] = u;
            psi[i + 1] = (&s * &v)?.with(())?;
        }

        // do same as rev order
    }
    Ok(())
}

// fn test_simple_code() {
//     let left = Tensor::zero([]);
//     for j in 0..(i - 1) {
//         left = (&left * &psi[j])? << &blas?; // idea: use ref as a suger of .view(). Ctx always consume tensor, but tensor could be ref repr.
//         left = (&left * &hamil[j])? << &blas?;
//         left = (&left
//             * &psi[j]
//                 .conj()?
//                 .with(())?
//                 .replace(todo!("add prime to all legs")))?
//             << &blas?;
//         /* e.g. this chain seems easy to make mistake, but not.
//         1. tensor contraction (A*B) is only defined for view * view. so the second operand must be a view.
//         2. we cannot move psi[j], so we must take view.
//         3. now we can consume view to genrated new view, which is internally memoed as conjugated.

//         if delayed conj not implemented, view.conj() is not callable (precisely, no Ctx satisfy with(_)) condition).
//         in this case psi[j].clone().conj()?.with(())? is the way, but this is apparently less efficient.
//         */
//     }

//     let right = Tensor::zero([]);
//     for j in ((i + 2)..n).rev() {
//         right = (&right * &psi[j])? << &blas?;
//         right = (&right * &hamil[j])? << &blas?;
//         right = (&right
//             * psi[j]
//                 .view()
//                 .conj()?
//                 .with(())?
//                 .replace(todo!("add prime to all legs")))?
//             << &blas?;
//     }

//     {
//         let x = (&a * &b)?;
//     } << &blas;

//     let center = (&psi[i] * &psi[i + 1])? << &blas?;

//     // optimizer using left,right,center,hamil[i],hamil[i+1]

//     let (u, s, v) = &center.svd()? << &blas?;
//     psi[i] = u;
//     psi[i + 1] = (&s * &v)? << &blas?;
// }
