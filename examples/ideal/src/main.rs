use tensory_core::{
    basic::leg::{Id128, Prime, Tag},
    dense::DenseTensor,
    lm, v,
};

type Leg = Prime<Id128>;

fn main() {
    // this is an ideal example of tensory.

    // resource allocation handler
    let alloc = CoreAlloc::new(); // some memory allocator. This is necessary considering cache with multicore, NUMA architecture, MPI, etc.
    let handle = BlasHandle::new::<f64>(&alloc); // blas handler. dynamic load (not link) is reccommended for portability, but should selectable by user. use `alloc` for internal memory allocation, and try reuse it.

    // leg ids
    let a = Leg::new();
    let b = Leg::new();
    let c = Leg::new();
    let d = Leg::new();
    let e = Leg::new();
    let f = Leg::new();

    let tag = Tag::from_raw(String::from("tag1"));

    // leg sizes: here we use usize for normal numeric tensors, but other TensorRepr can use other types.
    let a_n = 3;
    let b_n = 4;
    let c_n = 5;
    let d_n = 6;
    let e_n = 7;
    let f_n = 8;

    let rng = SmallRng::seed_from_u64(0);

    // in
    let ta = DenseTensor::random_using(lm![a=>a_n, b=>b_n, c=>c_n, d=>d_n], alloc, &mut rng); // f64 random tensor, allocated by `alloc`, with leg sizes specified by `lm![]` macro, and random number generator `rng`.
    let tb = DenseTensor::random_using(lm![c=>c_n, d=>d_n, e=>e_n, f=>f_n], alloc, &mut rng); // same as above, but with different leg sizes.
    // out
    let tc = DenseTensor::uninitialized(lm![a=>a_n, b=>b_n, e=>e_n, f=>f_n], alloc); // uninitialized f64 tensor, allocated by `alloc`, with leg sizes specified by `lm![]` macro.

    // delay contraction slightly, and provide executor using `by()` method.
    let tx = (&ta * &tb) // a (virtual) task tensor, which is just holding the two tensor. (note that &ta and &tb are just references, so we do auto convertion to a view tensor.)
        .assign(tc)? // this is also a task tensor, representing the operation of contraction and assignment.
        .with(handle)?; // here we provide the executor, which is a blas handle. This will execute the contraction and assignment, and return the result tensor.

    // cublas example
    let cuhandle = CuBlasHandle::new();
    let cuta = CuTensor::trans(ta, cuhandle);
    let cutb = CuTensor::trans(tb, cuhandle);
    let cutc = CuTensor::trans(tc, cuhandle);
    let cutx = (cuta * cutb).assign(cutc)?.with(cuhandle)?;
}

fn main() {
    todo!()
}
