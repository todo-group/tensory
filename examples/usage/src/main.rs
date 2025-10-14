//use num_complex::Complex;
// use tensory::{
//     basic::leg::{Id128, Prime},
//     nd_dense::NdDenseTensor,
// };

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_linalg::svd::TensorSvdExt;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, NdRuntime};

use tensory_core::leg;

type Leg = Prime<Id128>;
type Tensor<E> = NdDenseTensor<E, VecMapper<Leg>>;

fn main() -> anyhow::Result<()> {
    // resource allocation aware
    //let alloc = CoreAlloc::new();
    //let handle = BlasHandle::new();

    // leg
    let a = Leg::new();
    let b = Leg::new();
    let c = Leg::new();
    let d = Leg::new();
    let e = Leg::new();
    let f = Leg::new();

    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;
    let e_n = 50;
    let f_n = 60;

    // {
    //     let ta = Tensor::<f64>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n])?;
    //     let tb = Tensor::<f64>::random(leg![b=>b_n, c=>c_n, d=>d_n, a=>a_n])?;
    //     let x = (&ta + &tb)?.with(())?;
    // }

    // allocation aware syntax
    let ta = Tensor::<f64>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n])?;
    let tb = Tensor::<f64>::random(leg![c=>c_n, d=>d_n, e=>e_n, f=>f_n])?;
    let tc = Tensor::<f64>::zero(leg![a=>a_n, b=>b_n, e=>e_n, f=>f_n])?;

    // let nd = NdRuntime;

    // let ta = ta.bind(&nd);
    // let tb = tb.bind(&nd);

    // let tx = (&ta * &tb)?;

    let us_leg = Leg::new();
    let vs_leg = Leg::new();

    let mut x = Duration::ZERO;

    let d = 20;
    for t in 0..1 {
        println!("t = {}", t);

        let tx = Tensor::<f64>::random(leg![ b=>b_n, e=>e_n,a=>a_n, f=>f_n]).unwrap();

        let svd = (&tx).svd(leg![&a, &b], us_leg, vs_leg)?;

        let pre = Instant::now();

        black_box(svd.with(((),)))?;

        let post = Instant::now();

        x += post.duration_since(pre);
    }

    println!("{}", x.as_secs_f64() / 100.0);

    //svd(A, [a, b, c], us, sv);

    // let cuhandle = CuBlasHandle::new();
    // let cuta = CuTensor::trans(ta);
    // let cutb = CuTensor::trans(tb);
    // let cutc = CuTensor::trans(tc);
    // let cutx = (cuta * cutb).by(handle.mul(cutc));

    Ok(())
}
