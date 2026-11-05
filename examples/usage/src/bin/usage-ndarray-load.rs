use std::time::Instant;

use ndarray::Array2;
use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt};

type Tensor<'a, E> = NdDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    // the same workload as `usage-cuda-load` (examples/cuda), run on the CPU with tensory-ndarray.
    //
    // usage: usage-ndarray-load [size] [iterations]
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().map_or(Ok(4096), |s| s.parse())?;
    let iters: usize = args.next().map_or(Ok(20), |s| s.parse())?;
    let report_every = 5;

    // two n x n matrices (same contents as in the cuda example).
    let a_arr = Array2::from_shape_fn((n, n), |(i, j)| ((i * n + j) % 5) as f32);
    let b_arr = Array2::from_shape_fn((n, n), |(i, j)| ((i * n + j) % 7) as f32);
    let a = Tensor::<f32>::from_array(a_arr.clone().into_dyn(), ls!["i", "j"])?;
    let b = Tensor::<f32>::from_array(b_arr.clone().into_dyn(), ls!["j", "k"])?;

    // 2 n^3 flops per contraction.
    let flops_per_iter = 2.0 * (n as f64).powi(3);
    let start = Instant::now();
    let mut last = start;

    let mut c = (&a * &b)?.exec()?;
    for it in 0..iters {
        // d := (c + c) - c, c := a * b
        c = (&a * &b)?.exec()?;
        let c2 = (&c + &c)?.exec()?;
        let _d = (&c2 - &c)?.exec()?;

        if (it + 1) % report_every == 0 {
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

    // check one element of the last product.
    let expected: f32 = (0..n).map(|j| a_arr[[0, j]] * b_arr[[j, 0]]).sum();
    println!(
        "c[0, 0] = {} (expected {})",
        c.get(lm![&"i" => 0, &"k" => 0])??,
        expected
    );

    Ok(())
}
