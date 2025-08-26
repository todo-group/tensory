use num_complex::Complex;
use tensory::{
    basic::leg::{Id128, Prime},
    nd_dense::NdDenseTensor,
};
type Leg = Prime<Id128>;

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

    // allocation aware syntax
    let ta = NdDenseTensor::<_, Complex<f64>>::zero(leg![a=>1, b=>2, c=>3, d=>4].unwrap());
    let tb = NdDenseTensor::zero(leg![c=>3, d=>4, e=>5, f=>6].unwrap());
    //let tc = NdDenseTensor::zero(v![a=>1, b=>2, e=>5, f=>6]);

    let tx = (ta.view() * tb.view()).with(())?;

    let us_leg = Leg::new();
    let vs_leg = Leg::new();

    let (_u, _s, _v) = tx
        .view()
        .svd(leg_ref![&a, &b].unwrap(), us_leg, vs_leg)?
        .with(())?;

    //svd(A, [a, b, c], us, sv);

    // let cuhandle = CuBlasHandle::new();
    // let cuta = CuTensor::trans(ta);
    // let cutb = CuTensor::trans(tb);
    // let cutc = CuTensor::trans(tc);
    // let cutx = (cuta * cutb).by(handle.mul(cutc));

    Ok(())
}
