//use num_complex::Complex;
// use tensory::{
//     basic::leg::{Id128, Prime},
//     nd_dense::NdDenseTensor,
// };

use tensory_basic::{
    broker::VecBroker,
    id::{Id128, Prime},
};
use tensory_linalg::svd::TensorSvdExt;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt};

use tensory_core::leg;

type Leg = Prime<Id128>;
type Tensor<E> = NdDenseTensor<E, VecBroker<Leg>>;

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

    // allocation aware syntax
    let ta = Tensor::<f64>::random(leg![a=>a_n, b=>b_n, c=>c_n, d=>d_n]).unwrap();
    let tb = Tensor::random(leg![c=>c_n, d=>d_n, e=>e_n, f=>f_n]).unwrap();
    //let tc = Tensor::zero(v![a=>1, b=>2, e=>5, f=>6]);

    let tx = (&ta * &tb)?.with(())?;

    let us_leg = Leg::new();
    let vs_leg = Leg::new();

    let tx = Tensor::<f64>::random(leg![a=>30, b=>30, e=>30, f=>30]).unwrap();

    let svd = tx.view().svd(leg![&a, &b], us_leg, vs_leg)?;

    let pre = chrono::Local::now();

    let (_u, _s, _v) = svd.with(((),))?;

    let post = chrono::Local::now();

    println!("{}", post - pre);

    //svd(A, [a, b, c], us, sv);

    // let cuhandle = CuBlasHandle::new();
    // let cuta = CuTensor::trans(ta);
    // let cutb = CuTensor::trans(tb);
    // let cutc = CuTensor::trans(tc);
    // let cutx = (cuta * cutb).by(handle.mul(cutc));

    Ok(())
}
