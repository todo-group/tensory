//! cudarc を用いた行列積の素朴な実装 (start up 用)。行列は全て row-major。

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

type Result<T> = core::result::Result<T, Box<dyn std::error::Error>>;

const MATMUL_SRC: &str = r#"
extern "C" __global__ void matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
}
"#;

/// `a`: m x k, `b`: k x n -> m x n。NVRTC でコンパイルした自前カーネルで計算する。
pub fn matmul_kernel(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let func = ctx
        .load_module(compile_ptx(MATMUL_SRC)?)?
        .load_function("matmul")?;

    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

    const BLOCK: u32 = 16;
    let cfg = LaunchConfig {
        block_dim: (BLOCK, BLOCK, 1),
        grid_dim: ((n as u32).div_ceil(BLOCK), (m as u32).div_ceil(BLOCK), 1),
        shared_mem_bytes: 0,
    };
    let (m, n, k) = (m as i32, n as i32, k as i32);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&a_dev)
        .arg(&b_dev)
        .arg(&mut c_dev)
        .arg(&m)
        .arg(&n)
        .arg(&k);
    unsafe { builder.launch(cfg) }?;

    Ok(stream.clone_dtoh(&c_dev)?)
}

/// `a`: m x k, `b`: k x n -> m x n。cuBLAS の sgemm で計算する。
pub fn matmul_cublas(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<f32>(m * n)?;

    // cuBLAS は column-major なので、C^T = B^T A^T として row-major の A, B をそのまま渡す。
    let cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0f32,
        lda: n as i32,
        ldb: k as i32,
        beta: 0.0f32,
        ldc: n as i32,
    };
    unsafe { blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev) }?;

    Ok(stream.clone_dtoh(&c_dev)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: usize = 2;
    const K: usize = 3;
    const N: usize = 4;
    const A: [f32; M * K] = [1., 2., 3., 4., 5., 6.];
    const B: [f32; K * N] = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    const C: [f32; M * N] = [38., 44., 50., 56., 83., 98., 113., 128.];

    // GPU が必要
    #[test]
    #[ignore]
    fn kernel() {
        assert_eq!(matmul_kernel(&A, &B, M, K, N).unwrap(), C);
    }

    #[test]
    #[ignore]
    fn cublas() {
        assert_eq!(matmul_cublas(&A, &B, M, K, N).unwrap(), C);
    }
}
