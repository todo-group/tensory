// use tensory_core::{
//     basic::leg::{Id128, Prime, Tag},
//     dense::DenseTensor,
//     v,
// };

// type Leg = Prime<Id128>;

// fn main() {
//     // this is an ideal example of tensory.

//     // resource allocation handler
//     let alloc = CoreAlloc::new(); // some memory allocator. This is necessary considering cache with multicore, NUMA architecture, MPI, etc.
//     let handle = BlasHandle::new(); // blas handler. dynamic load (not link) is reccommended for portability.

//     // leg ids
//     let a = Leg::new();
//     let b = Leg::new();
//     let c = Leg::new();
//     let d = Leg::new();
//     let e = Leg::new();
//     let f = Leg::new();

//     let tag = Tag::from_raw(String::from("tag1"));

//     // leg sizes: here we use usize for normal numeric tensors, but other TensorRepr can use other types.
//     let a_n = 3;
//     let b_n = 4;
//     let c_n = 5;
//     let d_n = 6;
//     let e_n = 7;
//     let f_n = 8;

//     // in
//     let ta = DenseTensor::random(v![a=>a_n, b=>b_n, c=>c_n, d=>d_n], alloc);
//     let tb = DenseTensor::random(v![c=>c_n, d=>d_n, e=>e_n, f=>f_n], alloc);
//     // work, for transpose
//     let ta_work = DenseTensor::zero(v![a=>a_n, b=>b_n, c=>c_n, d=>d_n], alloc);
//     let tb_work = DenseTensor::zero(v![c=>c_n, d=>d_n, e=>e_n, f=>f_n], alloc);
//     // out
//     let tc = DenseTensor::zero(v![a=>a_n, b=>b_n, e=>e_n, f=>f_n], alloc);

//     // delay contraction slightly, and provide executor using `by()` method.
//     let tx = (ta * tb).with((handle, ta_work, tb_work, tc))?;

//     // cublas example
//     let cuhandle = CuBlasHandle::new();
//     let cuta = CuTensor::trans(ta, cuhandle);
//     let cutb = CuTensor::trans(tb, cuhandle);
//     let cutc = CuTensor::trans(tc, cuhandle);
//     let cutx = (cuta * cutb).with((cuhandle, cutc));
// }

fn main() {
    todo!()
}
