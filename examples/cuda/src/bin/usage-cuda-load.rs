use std::time::Instant;

use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_cuda::{CudaDenseTensor, CudaDenseTensorExt, CudaRuntime};

type Tensor<'a, E> = CudaDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    // a heavy variant of `usage-cuda`: the same simple operations on large tensors, repeated in a loop,
    // so that the load is visible in `nvidia-smi` (e.g. `watch -n 0.5 nvidia-smi`).
    //
    // usage: usage-cuda-load [size] [iterations]
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(Ok(4096), |s| s.parse())?;
    let iters: usize = args.next().map_or(Ok(1000), |s| s.parse())?;
    let report_every = 100;

    let rt = CudaRuntime::new(0)?;

    // two n x n matrices (4096 x 4096 x 4 bytes = 64 MiB each by default), kept on the device during the loop.
    let a_host: Vec<f32> = (0..n * n).map(|x| (x % 5) as f32).collect();
    let b_host: Vec<f32> = (0..n * n).map(|x| (x % 7) as f32).collect();
    let a = Tensor::<f32>::from_host(rt.stream(), lm!["i"=>n, "j"=>n], &a_host)?;
    let b = Tensor::<f32>::from_host(rt.stream(), lm!["j"=>n, "k"=>n], &b_host)?;

    // 2 n^3 flops per contraction.
    let flops_per_iter = 2.0 * (n as f64).powi(3);
    let start = Instant::now();
    let mut last = start;

    let mut c = (&a * &b)?.with(rt.clone())?;
    for it in 0..iters {
        // d := (c + c) - c, c := a * b
        // the calls only enqueue work on the stream; the host runs ahead of the device.
        c = (&a * &b)?.with(rt.clone())?;
        let c2 = (&c + &c)?.with(rt.clone())?;
        let _d = (&c2 - &c)?.with(rt.clone())?;

        if (it + 1) % report_every == 0 {
            // wait for the device so that the reported throughput is real.
            rt.stream().synchronize()?;
            let now = Instant::now();
            println!(
                "iteration {:>6}/{}: {:.1} GFLOP/s, elapsed {:.1?}",
                it + 1,
                iters,
                flops_per_iter * report_every as f64 / (now - last).as_secs_f64() / 1e9,
                now - start
            );
            last = now;
        }
    }
    rt.stream().synchronize()?;

    // check one element of the last product on the host.
    let c_host = c.repr().to_host()?;
    let expected: f32 = (0..n).map(|j| a_host[j] * b_host[j * n]).sum();
    println!("c[0, 0] = {} (expected {})", c_host[0], expected);

    Ok(())
}
