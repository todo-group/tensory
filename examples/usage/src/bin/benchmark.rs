//use num_complex::Complex;
// use tensory::{
//     basic::leg::{Id128, Prime},
//     nd_dense::NdDenseTensor,
// };

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use itertools::{Itertools, min};
use rand::{SeedableRng, rngs::SmallRng};
use tensory_basic::mapper::VecMapper;
use tensory_linalg::prelude::*;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt};

use tensory_core::prelude::*;

type Tensor<'a, E> = NdDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    // this is an example of tensory-ndarray svd benchmark.

    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;

    let mut rng = SmallRng::seed_from_u64(0);
    let tx =
        Tensor::<f64>::random_using(lm!["a"=>a_n, "b"=>b_n, "c"=>c_n, "d"=>d_n], &mut rng).unwrap();

    let legs = ["a", "b", "c", "d"];
    let sizes = [a_n, b_n, c_n, d_n];

    let repeat: usize = 100;

    for perm in (0..4).permutations(4) {
        let [(a_x, a_x_n), (b_x, b_x_n), (c_x, c_x_n), (d_x, d_x_n)] = (0..4)
            .map(|i| (legs[perm[i]], sizes[perm[i]]))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let h = a_x_n * b_x_n;
        let w = c_x_n * d_x_n;

        let order_expected = h * w * min([h, w]).unwrap();

        let mut t_tmp = Duration::ZERO;

        for _ in 0..repeat {
            let pre = Instant::now();
            black_box((&tx).svd(ls![&a_x, &b_x], "e", "f")?.with(((),)).unwrap());
            let post = Instant::now();
            t_tmp += post.duration_since(pre);
        }

        let time_avg = t_tmp.as_secs_f64() / repeat as f64;

        println!(
            "perm: [{:?} {:?} {:?} {:?}], avg time: {:.6} s, expected order: {:.6}, ratio: {:.6} s",
            a_x,
            b_x,
            c_x,
            d_x,
            time_avg,
            order_expected as f64 * 1e-9,
            time_avg / (order_expected as f64 * 1e-9),
        );
    }

    Ok(())
}
