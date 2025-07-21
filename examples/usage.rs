use tensory::{
    basic::leg::{Id128, Prime},
    nd_dense::NdDenseTensor,
    v,
};
type Index = Prime<Id128>;

fn main() {
    /// resource allocation aware
    //let alloc = CoreAlloc::new();
    //let handle = BlasHandle::new();

    /// index
    let a = Index::new();
    let b = Index::new();
    let c = Index::new();
    let d = Index::new();
    let e = Index::new();
    let f = Index::new();

    /// allocation aware syntax
    let ta = NdDenseTensor::<Index, f64>::zero(v![a=>1, b=>2, c=>3, d=>4]);
    let tb = NdDenseTensor::zero(v![c=>3, d=>4, e=>5, f=>6]);
    //let tc = NdDenseTensor::zero(v![a=>1, b=>2, e=>5, f=>6]);

    let tx = (ta.view() * tb.view()).by(());

    // let cuhandle = CuBlasHandle::new();
    // let cuta = CuTensor::trans(ta);
    // let cutb = CuTensor::trans(tb);
    // let cutc = CuTensor::trans(tc);
    // let cutx = (cuta * cutb).by(handle.mul(cutc));
}
