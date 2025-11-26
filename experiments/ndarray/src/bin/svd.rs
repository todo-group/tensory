use ndarray::{Array1, Array2};
use ndarray_linalg::{JobSvd, SVD, SVDDC, TruncatedOrder, TruncatedSvd, random_using};
use num_traits::ToPrimitive;
use rand::{SeedableRng, rngs::SmallRng};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

extern crate blas_src as _;

fn measure_svd_time(
    h: usize,
    w: usize,
    trials: usize,
    svd_func: impl Fn(Array2<f64>) -> (Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>),
) -> f64 {
    let mut total_time = Duration::ZERO;

    let mut rng = SmallRng::seed_from_u64(0);

    for _ in 0..trials {
        // Create random matrix
        let x: Array2<f64> = random_using([h, w], &mut rng);

        let pre = Instant::now();

        black_box(svd_func(x));

        let post = Instant::now();

        total_time += post.duration_since(pre);
    }

    total_time.as_secs_f64() / trials.to_f64().unwrap()
}

fn compare_svd(h: usize, w: usize, trials: usize) {
    println!(
        "{} x {} matrix SVD comparison over {} trials:",
        h, w, trials
    );

    let svd_full = measure_svd_time(h, w, trials, |x| x.svd(true, true).unwrap());
    println!("SVD Full took {:.4} seconds", svd_full);

    let sdd_thin = measure_svd_time(h, w, trials, |x| x.svddc(JobSvd::Some).unwrap());
    println!("SDD Thin took {:.4} seconds", sdd_thin);

    let sdd_full = measure_svd_time(h, w, trials, |x| x.svddc(JobSvd::All).unwrap());
    println!("SDD Full took {:.4} seconds", sdd_full);

    let tsvd = measure_svd_time(h, w, trials, |x| {
        let (u, d, v) = TruncatedSvd::new(x, TruncatedOrder::Largest)
            .decompose(5)
            .unwrap()
            .values_vectors();
        (Some(u), d, Some(v))
    });
    println!("Truncated SVD took {:.4} seconds", tsvd);

    println!();
}

fn main() {
    compare_svd(400, 400, 100);
    compare_svd(200, 3000, 100); // 100 might be too long for large matrices
}
