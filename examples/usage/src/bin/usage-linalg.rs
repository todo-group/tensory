use num_complex::Complex;
use rand::{SeedableRng, rngs::SmallRng};
use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_core::prelude::*;
use tensory_linalg::prelude::*;
use tensory_ndarray::{
    NdDenseTensor, NdDenseTensorExt,
    linalg::{DiagExp, DiagPow, DiagPowI, Half, HermiteEig},
};

// type aliases for convenience. You can change them to other implementations.
type Leg = Prime<Id128>;
type Tensor<E> = NdDenseTensor<E, VecMapper<Leg>>;

fn main() -> anyhow::Result<()> {
    // legs
    let a = Leg::new();
    let b = Leg::new();
    let c = Leg::new();
    let d = Leg::new();
    let e = Leg::new();
    let f = Leg::new();

    // leg size
    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;

    let mut rng = SmallRng::seed_from_u64(0);

    // norm
    {
        let t = Tensor::<f64>::random_using(lm![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?; // [a,b,c,d]
        let n = (&t).norm().exec()?; // scalar
        println!("norm: {}", n);
    }

    // pow (sqrt)
    {
        let mut t = Tensor::<f64>::zero(lm![a=>a_n,a.prime()=>a_n,b=>b_n,b.prime()=>b_n])?; // [a,a',b,b']
        for ai in 0..a_n {
            for bi in 0..b_n {
                t[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]] = (ai + bi) as f64;
            }
        }

        let t_exp = (&t)
            .pow(Half, ls![(&a, &a.prime()), (&b, &b.prime())])?
            .exec()?; // [a,b,a',b']

        let mut sum = 0.0;
        for ai in 0..a_n {
            for bi in 0..b_n {
                sum += (t_exp[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]]
                    - ((ai + bi) as f64).sqrt())
                .abs();
            }
        }
        println!("sqrt error: {}", sum);
    }

    // exp
    {
        let mut t = Tensor::<f64>::zero(lm![a=>a_n,a.prime()=>a_n,b=>b_n,b.prime()=>b_n])?; // [a,a',b,b']
        for ai in 0..a_n {
            for bi in 0..b_n {
                t[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]] = (ai + bi) as f64;
            }
        }

        let t_exp = (&t)
            .exp(ls![(&a, &a.prime()), (&b, &b.prime())])?
            .with(DiagExp)?; // [a,b,a',b']

        let mut sum = 0.0;
        for ai in 0..a_n {
            for bi in 0..b_n {
                sum += (t_exp[lm![&a=>ai,&a.prime()=>ai,&b=>bi,&b.prime()=>bi]]
                    - ((ai + bi) as f64).exp())
                .abs();
            }
        }
        println!("exp error: {}", sum);
    }

    // conj
    {
        let t = Tensor::<Complex<f64>>::random_unitary_using(
            lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n],
            &mut rng,
        )?; // [a,a',b,b']
        let t_conj = (&t).conj().exec()?; // [a,a',b,b']

        let t_conj = t_conj.replace_leg(lm![&a=>a.prime_by(2),&b=>b.prime_by(2)])?; // [a'',a',b'',b']

        let eye_from_t = (&t * &t_conj)?.exec()?; // [a,a'',b,b'']

        let eye = Tensor::<Complex<f64>>::eye(lm![[a,a.prime_by(2)]=>a_n, [b,b.prime_by(2)]=>b_n])?; // [a,a'',b,b'']

        println!(
            "relative error of identity from unitary (I ?= U * U^†): {}",
            (&(&eye_from_t - &eye)?.exec()?).norm().exec()? / (&t).norm().exec()?
        );
    }

    // svd
    {
        let t = Tensor::<f64>::random_using(lm![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?; // [a,b,c,d]
        let (u, s, v_dagger) = (&t).svd(ls![&a, &b], e, f)?.exec()?; // u:[a,b,e], s:[e,f], v:[f,c,d]
        let t_reconstructed = (&(&u * &s)?.exec()? * &v_dagger)?.exec()?; // [a,b,c,d]
        println!(
            "relative error of eig reconstruct (A ?= U * S * V^†): {}",
            (&(&t_reconstructed - &t)?.exec()?).norm().exec()? / (&t).norm().exec()?
        );
    }

    // qr
    {
        let t = Tensor::<f64>::random_using(lm![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?; // [a,b,c,d]
        let (q, r) = (&t).qr(ls![&a, &b], e)?.with(())?; // q:[a,b,e], r:[e,c,d]
        let t_reconstructed = (&q * &r)?.exec()?; // [a,b,c,d]
        println!(
            "relative error of qr reconstruct (A ?= Q * R): {}",
            (&(&t_reconstructed - &t)?.exec()?).norm().exec()? / (&t).norm().exec()?
        );
    }

    // eig
    {
        let t = Tensor::<f64>::random_hermite_using(
            lm![[a,a.prime()]=>a_n, [b,b.prime()]=>b_n],
            &mut rng,
        )?; // [a,a',b,b']
        let (v, d, v_dagger) = (&t)
            .eig(ls![(&a, &a.prime()), (&b, &b.prime())], e, f)?
            .with(HermiteEig)?; // v:[a,a',e], d:[e,f], vc:[f,b',b]
        let t_reconstructed = (&(&v * &d)?.exec()? * &v_dagger)?.exec()?; // [a,a',b,b']
        println!(
            "relative error of eig reconstruct (A ?= V * D * V^†): {}",
            (&(&t_reconstructed - &t)?.exec()?).norm().exec()? / (&t).norm().exec()?
        );
    }

    // solve eig
    {
        let t = Tensor::<Complex<f64>>::random_using(
            lm![a => a_n, a.prime() => a_n, b => b_n, b.prime() => b_n],
            &mut rng,
        )?; // [a,a',b,b']
        let (v, d) = (&t)
            .solve_eig(ls![(&a, &a.prime()), (&b, &b.prime())], e, f)?
            .with(())?; // v:[a,b,e], d:[e,f]
        let t_v = (
            &t * (&v).replace_leg(lm![&a=>a.prime(),&b=>b.prime(),&e=>f])?
            // [a',b',f]
        )?
        .exec()?; // [a,b,f]
        let v_d = (&v * &d)?.exec()?; // [a,b,f]
        println!(
            "relative error after eig solve (A * V ?= V * D): {}",
            (&(&v_d - &t_v)?.exec()?).norm().exec()? / (&t_v).norm().exec()?
        );
    }

    Ok(())
}
