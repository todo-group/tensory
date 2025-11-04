use num_complex::Complex;

use std::{
    f64::consts::PI,
    hint::black_box,
    time::{Duration, Instant},
};

use rand::{SeedableRng, rngs::SmallRng};
use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_linalg::{norm::TensorNormExt, svd::TensorSvdExt};
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, NdRuntime};

use tensory_core::{leg, tensor::TensorTask};

// type aliases for convenience. You can change them to other implementations.
type Leg = Prime<Id128>;
type Tensor<E> = NdDenseTensor<E, VecMapper<Leg>>;

fn main() -> anyhow::Result<()> {
    // first, we define legs.
    let a = Leg::new();
    let b = Leg::new();
    let c = Leg::new();
    let d = Leg::new();
    let e = Leg::new();
    let f = Leg::new();

    // then we decides dimensions for each leg.
    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;
    let e_n = 50;
    let f_n = 60;

    // use fast and seedable random number generator (RNG) for speed and reproducibility.
    let mut rng = SmallRng::seed_from_u64(0);

    println!("random t");

    // here we create a random tensor with legs a,b,c,d,e,f, using `rng` as RNG.
    // each leg is assigned with its dimension.
    // (note: this constructor is provided by `NdDenseTensorExt` trait)
    let t = Tensor::<f64>::random_using(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?;

    println!("done");

    // alternatively, you can use `Tensor::random(...)` method to create a random tensor using default RNG.
    // let t = Tensor::<f64>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n, e=>e_n, f=>f_n])?;

    // scalar multiplication and division using the overloaded `*` and `/` operator.
    // here we use strategy pattern-like api to perform the actual computation using `.with(())` method.
    // same pattern appears in many other operations in tensory. The middle struct implements TensorTask<C> trait.
    let t_mul_pi = (t.clone() * PI).with(())?;
    let t_mul_pi_div_pi = (t_mul_pi / PI).with(())?;

    // due to the limitation of rust syntax, tuple is required for scalar operations. but if the scalar implements TensorScalar, it works without tuple in right mul/div.
    // let t_double = (t * (2.0,)).with(())?;
    // let t_double_half = (t_double / (2.0,)).with(())?;

    let diff = (&t_mul_pi_div_pi - &t)?.with(())?;

    println!("difference norm: {}", (&diff).norm().with(())?);

    let t_2 = Tensor::<f64>::random_using(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?;

    let t_sum = (&t + &t_2)?.with(())?;

    // // {
    // //     let ta = Tensor::<f64>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n])?;
    // //     let tb = Tensor::<f64>::random(leg![b=>b_n, c=>c_n, d=>d_n, a=>a_n])?;
    // //     let x = (&ta + &tb)?.with(())?;
    // // }

    // // allocation aware syntax

    // println!("let's go");

    // let ta = Tensor::<f64>::random_using(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n], &mut rng)?;
    // let tb = Tensor::<f64>::random_using(leg![c=>c_n, d=>d_n, e=>e_n, f=>f_n], &mut rng)?;
    // let tc = Tensor::<f64>::zero(leg![a=>a_n, b=>b_n, e=>e_n, f=>f_n])?;

    // println!("before mul");

    // let tx = (&ta * &tb)?.with(())?;

    // println!("after mul");

    // let nd = NdRuntime;

    // let ta = ta.bind(nd);
    // let tb = tb.bind(nd);

    // let tx = (&ta * &tb)?;

    // let us_leg = Leg::new();
    // let vs_leg = Leg::new();

    // let mut x = Duration::ZERO;

    // let d = 20;
    // let turns: usize = 100;

    // for t in 0..turns {
    //     println!("t = {}", t);

    //     let tx =
    //         Tensor::<f64>::random_using(leg![ b=>b_n, e=>e_n,a=>a_n, f=>f_n], &mut rng).unwrap();

    //     let svd = (&tx).svd(leg![&a, &b], us_leg, vs_leg)?;

    //     let pre = Instant::now();

    //     black_box(svd.with(((),)))?;

    //     let post = Instant::now();

    //     x += post.duration_since(pre);
    // }

    // println!("{}", x.as_secs_f64() / turns as f64);

    // //svd(A, [a, b, c], us, sv);

    // // let cuhandle = CuBlasHandle::new();
    // // let cuta = CuTensor::trans(ta);
    // // let cutb = CuTensor::trans(tb);
    // // let cutc = CuTensor::trans(tc);
    // // let cutx = (cuta * cutb).by(handle.mul(cutc));

    Ok(())
}
