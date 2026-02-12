use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_core::prelude::*;
use tensory_linalg::prelude::*;
use tensory_ndarray::{
    NdDenseTensor, NdDenseTensorExt,
    cut_filter::max_ix,
    regulated::{L2Regulator, LnCoeff},
};
use tensory_regulated::{
    RegulatedTensorExt, TensorDefaultRegulatedTask, TensorRegulatedTask, ToRegulatedTensorExt,
};

type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;
fn main() -> anyhow::Result<()> {
    let t: usize = 80;
    let d: usize = 20;

    let mut x = Leg::new();
    let mut y = Leg::new();

    let j: f64 = 1.0;
    let h: f64 = 0.0;
    let beta: f64 = 0.381;

    let mut z = Tensor::zero(lm![x=> 2, y=> 2, x.prime()=> 2, y.prime()=> 2]).unwrap();
    for x_i in 0..2 {
        for y_i in 0..2 {
            for x_p_i in 0..2 {
                for y_p_i in 0..2 {
                    let s_x = ((x_i as isize) * 2 - 1) as f64;
                    let s_xd = ((x_p_i as isize) * 2 - 1) as f64;
                    let s_y = ((y_i as isize) * 2 - 1) as f64;
                    let s_yd = ((y_p_i as isize) * 2 - 1) as f64;
                    z[lm![
                        &x=>x_i,
                        &y=>y_i,
                        &x.prime()=>x_p_i,
                        &y.prime()=>y_p_i
                    ]] = (beta
                        * (h * (s_x + s_xd + s_y + s_yd) / 2.0
                            + j * (s_x * s_y + s_y * s_xd + s_xd * s_yd + s_yd * s_x)))
                        .exp();
                }
            }
        }
    }
    let mut z = z.regulate::<LnCoeff<f64>, L2Regulator<_>>();

    for i in 0..t {
        std::println!("loop={}", i);

        let x_new = Leg::new();
        let y_new = Leg::new();

        let dum = Leg::new();

        let (u, s, v) = (z
            .view()
            .replace_leg(lm![&x => x.prime(),&y => y.prime(), &x.prime() => x, &y.prime() => y])
            .unwrap())
        .svd_with_more_ids(ls![&x, &y], x_new, x_new.prime(), dum, dum)?
        .with_regulated(max_ix(d))?;
        let A = u;
        let B = (&s * &v)?.exec()?;

        let (u, s, v) = (&z)
            .svd_with_more_ids(ls![&x, &y.prime()], y_new, y_new.prime(), dum, dum)?
            .with_regulated(max_ix(d))?;

        let C = u;
        let D = (&s * &v)?.exec()?;

        // bound tensor should provide common single runtime const reference
        // and options are passed in first call fn
        // then convert it to the pair of runtime and option
        // this tuple should impl XxxCtx

        z = (&(&A * &D)?.exec()? * &(&B * &C)?.exec()?)?.exec()?;

        x = x_new;
        y = y_new;

        let eye_x = Tensor::eye(lm![[x,x.prime()]=>z.axis_info(&x).unwrap()])?.regulate();
        let eye_y = Tensor::eye(lm![[y,y.prime()]=>z.axis_info(&y).unwrap()])?.regulate();

        let z_core = (&(&z * &eye_x)?.exec()? * &eye_y)?.exec()?;
        let (z_core, coeff) = z_core.into_inner_tensor();

        std::println!(
            "Log partition function: {}",
            (coeff.ln() + z_core[lm![]].ln()) / 2.0f64.powi((i + 2) as i32)
        );
    }

    Ok(())
}

// rec,d,j,h,beta,lnz,f,e,c
// 11,20,1,0,0.381,0.8592707521119247,-2.2553038113173876,-1.0112731075256898,0.6746082947341026
// 11,20,1,0,0.555,1.1254365827897135,-2.027813662684168,-1.8585112245461666,0.4411754976141325
