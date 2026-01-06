use num_complex::Complex;
use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_core::prelude::*;
use tensory_linalg::svd::TensorSvdExt;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, cut_filter::max_ix};

type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;
fn main() -> anyhow::Result<()> {
    // Example usage of the negation operation
    let step: usize = 100;
    let d: usize = 10;
    let temperature: f64 = 2.0;
    let beta = 0.5 * ((2.0 / temperature).exp() + ((4.0 / temperature).exp() - 1.0).sqrt()).ln();
    println!("temperature = {}", temperature);

    //println!("beta = {}", beta);

    let l = Leg::new();
    let t = Leg::new();
    let r = Leg::new();
    let b = Leg::new();

    let mut el: Leg = Leg::new();
    let mut er: Leg = Leg::new();
    let mut ea: Leg = Leg::new();

    let mut a = Tensor::zero(lm![l=> 2, t=> 2, r=> 2, b=> 2]).unwrap();
    let mut spin_a = Tensor::zero(lm![l=> 2, t=> 2, r=> 2, b=> 2]).unwrap();
    for l_i in 0..2 {
        let l_is = (l_i as f64 - 0.5) * 2.0;
        for t_i in 0..2 {
            let t_is = (t_i as f64 - 0.5) * 2.0;
            for r_i in 0..2 {
                let r_is = (r_i as f64 - 0.5) * 2.0;
                for b_i in 0..2 {
                    let b_is = (b_i as f64 - 0.5) * 2.0;

                    a[lm![
                        &l=>l_i,
                        &t=>t_i,
                        &r=>r_i,
                        &b=>b_i
                    ]] = 2.0 * ((l_is + t_is + r_is + b_is) * beta).cosh();
                    // impurity tensor
                    spin_a[lm![
                        &l=>l_i,
                        &t=>t_i,
                        &r=>r_i,
                        &b=>b_i
                    ]] = 2.0 * ((l_is + t_is + r_is + b_is) * beta).sinh();
                }
            }
        }
    }

    // normalization
    let dummy = Leg::new();
    let (u, s, v) = (&a)
        .svd(ls![&l, &t], dummy, dummy.prime())?
        .with(max_ix(4))?;
    let factor_initial = s[lm![&dummy=>0,&dummy.prime()=>0]]
        + s[lm![&dummy=>1,&dummy.prime()=>1]]
        + s[lm![&dummy=>2,&dummy.prime()=>2]]
        + s[lm![&dummy=>3,&dummy.prime()=>3]];

    // println!("a factor {}", factor_initial);

    a = (a / (factor_initial,)).with(())?;

    let mut c = Tensor::zero(lm![el=> 2, er=> 2]).unwrap();

    for el_i in 0..2 {
        let el_is = (el_i as f64 - 0.5) * 2.0;
        for er_i in 0..2 {
            let er_is = (er_i as f64 - 0.5) * 2.0;
            // ferro boundary condition
            // c[lm![&el=>el_i, &er=>er_i]] = ((el_is + er_is) * beta).exp();
            // open boundary condition
            c[lm![&el=>el_i, &er=>er_i]] = 2.0 * ((el_is + er_is) * beta).cosh();
        }
    }

    //normalization
    c = (c / (factor_initial,)).with(())?;

    let mut etn = Tensor::zero(lm![el=> 2, er=> 2, ea=>2]).unwrap();
    for el_i in 0..2 {
        let el_is = (el_i as f64 - 0.5) * 2.0;
        for er_i in 0..2 {
            let er_is = (er_i as f64 - 0.5) * 2.0;
            for ea_i in 0..2 {
                let ea_is = (ea_i as f64 - 0.5) * 2.0;
                // ferro boundary condition
                // etn[lm![&el=>el_i, &er=>er_i, &ea=>ea_i]] = ((el_is + er_is + ea_is) * beta).exp();
                // open boundary condition
                etn[lm![&el=>el_i, &er=>er_i, &ea=>ea_i]] =
                    2.0 * ((el_is + er_is + ea_is) * beta).cosh();
            }
        }
    }

    //normalization
    etn = (etn / (factor_initial,)).with(())?;

    // normalization
    // let dummy = Leg::new();
    // let (u, s, v) = (&c).view().svd(ls![&el], dummy, dummy.prime())?.with((MaxIx(1),))?;
    // let factor = s[ls![&dummy=>0,&dummy.prime()=>0]];
    // println!("initial factor {}", factor);

    // c = (c / (factor,)).with(())?;
    // etn = ((etn) / (factor.sqrt(),)).with(())?;

    let mut factors: Vec<f64> = Vec::with_capacity(step);
    let mut e_factors: Vec<f64> = Vec::with_capacity(step);
    let mut e_factor_sum: f64 = 0.0;

    for _rgstep in 0..step {
        // std::println!("order c {}", c.repr().naxes());
        // std::println!("order et {}", etn.repr().naxes());

        let ec = (&etn * &c.replace_leg(lm![&er => er.prime()]).unwrap())?.with(())?;
        let ece = (&ec
            .replace_leg(lm![&er => er.prime().prime(), &er.prime() => er, &ea => ea.prime()])
            .unwrap()
            * &etn)?
            .with(())?;
        let mat = (&ece.replace_leg(lm![&ea => l, &ea.prime() => t]).unwrap() * &a)?.with(())?;

        let el_new = Leg::new();
        let (u, s, v) = (&mat)
            .view()
            .svd(ls![&el, &b], el_new, el_new.prime())?
            .with(max_ix(d))?;

        c = (&((&mat * &u)?.with(())?)
            .replace_leg(lm![&er.prime().prime() => el, &r => b, &el_new => el_new.prime()])
            .unwrap()
            * &u)?
            .with(())?;
        etn = (&((&((&etn * &u)?.with(())?)
            .replace_leg(lm![&b => l, &ea => t])
            .unwrap()
            * &a)?
            .with(())?)
        .replace_leg(lm![
        &er => el,
        &b => ea,
        &r => b
        ])
        .unwrap()
            * &u.replace_leg(lm![&el_new => el_new.prime()]).unwrap())?
            .with(())?;

        el = el_new;
        er = el_new.prime();

        let factor = s[lm![&el_new=>0,&el_new.prime()=>0]];
        //println!("factor at {}, {}", _rgstep, factor);

        c = (c / (factor,)).with(())?;
        etn = ((etn) / (factor.sqrt(),)).with(())?;

        factors.push(factor.ln());
        e_factor_sum += 0.5 * factor.ln();
        e_factors.push(e_factor_sum);

        let cp = c.clone();
        let c2 = (&c * &cp.replace_leg(lm![&el => el.prime(), &er => el]).unwrap())?.with(())?;
        let c4 = (&c2 * &c2)?.with(())?;

        let L1 = 2 * (_rgstep + 2);
        let sum_f = c4[lm![]];

        let factors_sum: f64 = factors.iter().sum();
        let e_factors_sum: f64 = e_factors.iter().sum();
        let e_final = e_factors[_rgstep];
        let a_factor = (2.0 * (1.0 / temperature).exp()).ln();
        let free_energy_density = -temperature
            * (4.0 * (factors_sum + 2.0 * (e_factors_sum - e_final)) + sum_f.ln()
                - (2 * L1 * (L1 - 1)) as f64 * a_factor
                + (L1 * L1) as f64 * factor_initial.ln())
            / (L1 * L1) as f64;
        println!("Free energy density for L {}: {}", L1, free_energy_density);

        if _rgstep > 0 {
            let delta_e = e_factors[_rgstep] - e_factors[_rgstep - 1];
            let free_energy_density_infinite =
                -temperature * (delta_e - 2.0 * a_factor + factor_initial.ln());
            println!(
                "Free energy density for infinite system {}",
                free_energy_density_infinite
            );
        }
    }
    Ok(())
}
