use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_core::prelude::*;
use tensory_linalg::prelude::*;
use tensory_ndarray::{
    NdDenseTensor, NdDenseTensorExt,
    cut_filter::max_ix,
    linalg::{Diag, Half},
};

type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;
fn main() -> anyhow::Result<()> {
    let t: usize = 11;
    let d: usize = 20;

    let mut x = Leg::new();
    let mut y = Leg::new();

    let j: f64 = 1.0;
    let h: f64 = 0.0;
    // let beta: f64 = 0.381; //
    let beta: f64 = (1.0 + 2.0f64.sqrt()).ln() / 2.0;

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

    let z_norm = (&z).norm().exec()?;
    z = (z / z_norm).exec()?;
    let mut coeff_log = z_norm.ln();

    for _renorm in 0..t {
        std::println!("x {:?}", x);
        std::println!("y {:?}", y);
        std::println!("legs {:?}", z.mapper());
        std::println!("order {}", z.repr().naxes());

        let x_new = Leg::new();
        let y_new = Leg::new();

        let dum = Leg::new();

        let (u, s, v) = (z
            .view()
            .replace_leg(lm![&x => x.prime(),&y => y.prime(), &x.prime() => x, &y.prime() => y])
            .unwrap())
        .svd_with_more_ids_and_option(ls![&x, &y], dum.prime(), dum.prime(), dum, dum, max_ix(d))?
        .with(())?
        .unpack();
        let s_sqrt = (&s).pow(Half, ls![(&dum.prime(), &dum)])?.with(Diag)?;
        let a = (&u * &s_sqrt)?.exec()?; // dum -> x_new
        let b = (&s_sqrt * &v)?.exec()?; // dum.prime() -> x_new.prime()
        let a = a.replace_leg(lm![&dum => x_new])?;
        let b = b.replace_leg(lm![&dum.prime() => x_new.prime()])?;

        let (u, s, v) = (&z)
            .svd_with_more_ids_and_option(
                ls![&x, &y.prime()],
                dum.prime(),
                dum.prime(),
                dum,
                dum,
                max_ix(d),
            )?
            .with(())?
            .unpack();

        let s_sqrt = (&s).pow(Half, ls![(&dum.prime(), &dum)])?.with(Diag)?;

        let c = (&u * &s_sqrt)?.exec()?; // dum -> y_new
        let d = (&s_sqrt * &v)?.exec()?; // dum.prime() -> y_new.prime()
        let c = c.replace_leg(lm![&dum => y_new])?;
        let d = d.replace_leg(lm![&dum.prime() => y_new.prime()])?;

        let z_tmp = (&(&a * &d)?.exec()? * &(&b * &c)?.exec()?)?.exec()?;

        let z_norm = (&z_tmp).norm().exec()?;

        z = (z_tmp / z_norm).exec()?;

        x = x_new;
        y = y_new;

        coeff_log *= 2.0;
        coeff_log += z_norm.ln();

        println!("renormed norm {}", z_norm);
    }

    let d_x = z.axis_info(0, &x)?;
    let d_y = z.axis_info(0, &y)?;

    let eye_x = Tensor::eye(lm![[x,x.prime()]=>d_x])?;
    let eye_y = Tensor::eye(lm![[y,y.prime()]=>d_y])?;

    let z = (&(&z * &eye_x)?.exec()? * &eye_y)?.exec()?;

    coeff_log += (&z).norm().exec()?.ln();

    std::println!("Final tensor: {:?}", z[lm![]]);

    let body_size = 2.0f64.powi((t + 1) as i32);
    let log_z = coeff_log / body_size;
    std::println!("body size: {}", body_size);
    std::println!("Log partition function: {}", log_z);
    std::println!("free energy: {}", -log_z / beta);

    Ok(())
}

// rec,d,j,h,beta,lnz,f,e,c
// 11,20,1,0,0.381,0.85927075211192 47,-2.2553038113173876,-1.0112731075256898,0.6746082947341026
// 11,20,1,0,0.555,1.1254365827897135,-2.027813662684168,-1.8585112245461666,0.4411754976141325

// at singular
// -2.110001060878283
// -2.110005667037697189306$
