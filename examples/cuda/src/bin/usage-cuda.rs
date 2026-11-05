use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_cuda::{CudaDenseTensor, CudaDenseTensorExt, CudaRuntime};

// type alias for convenience. `CudaDenseTensor` is the cuda counterpart of `NdDenseTensor`.
type Tensor<'a, E> = CudaDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    // a simple usage of tensory-cuda:
    // c := a * b   (contract the leg "j", computed by cuTENSOR)
    // d := (c + c) - c
    // print(max|d - c|, max|c - <the same product on host>|)

    let i_n = 3;
    let j_n = 4;
    let k_n = 5;

    // `CudaRuntime` owns a cuda stream and a cuTENSOR handle. it is passed to `.with()` as the context.
    // 0 is the ordinal of the device, i.e. the first GPU.
    let rt = CudaRuntime::new(0)?;

    // tensors are created from row-major host data, uploaded to the device.
    let a_host: Vec<f32> = (0..i_n * j_n).map(|x| x as f32).collect();
    let b_host: Vec<f32> = (0..k_n * j_n).map(|x| (x % 7) as f32).collect();
    let a = Tensor::<f32>::from_host(rt.stream(), lm!["i"=>i_n, "j"=>j_n], &a_host)?;
    // the axis order may differ between the operands: legs are matched by their IDs.
    let b = Tensor::<f32>::from_host(rt.stream(), lm!["k"=>k_n, "j"=>j_n], &b_host)?;

    // contraction. the result has legs "i", "k" (the remaining legs of `a`, then those of `b`).
    let c = (&a * &b)?.with(rt.clone())?;

    // addition and subtraction.
    let c2 = (&c + &c)?.with(rt.clone())?;
    let d = (&c2 - &c)?.with(rt)?;

    // download the results in logical row-major order (axes "i", "k") and compare on the host.
    let c_host = c.repr().to_host()?;
    let d_host = d.repr().to_host()?;

    let mut c_expected = vec![0f32; i_n * k_n];
    for i in 0..i_n {
        for k in 0..k_n {
            for j in 0..j_n {
                c_expected[i * k_n + k] += a_host[i * j_n + j] * b_host[k * j_n + j];
            }
        }
    }

    let max_diff = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    println!(
        "max|c - c_host| = {}, max|d - c| = {}",
        max_diff(&c_host, &c_expected),
        max_diff(&d_host, &c_host)
    );

    Ok(())
}
