use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, result::CublasError, sys::cublasOperation_t},
    driver::{CudaSlice, DeviceRepr, ValidAsZeroBits},
};

/// Element types supported by the cuda implementation.
pub trait CudaScalar: DeviceRepr + ValidAsZeroBits + Copy + Unpin + 'static {
    /// C type name used in NVRTC kernels.
    const C_NAME: &'static str;

    /// Row-major `c (m x n) = a (m x k) * b (k x n)`.
    fn gemm(
        blas: &CudaBlas,
        m: usize,
        n: usize,
        k: usize,
        a: &CudaSlice<Self>,
        b: &CudaSlice<Self>,
        c: &mut CudaSlice<Self>,
    ) -> Result<(), CublasError>;
}

macro_rules! impl_cuda_scalar {
    ($t:ty, $c:literal) => {
        impl CudaScalar for $t {
            const C_NAME: &'static str = $c;

            fn gemm(
                blas: &CudaBlas,
                m: usize,
                n: usize,
                k: usize,
                a: &CudaSlice<Self>,
                b: &CudaSlice<Self>,
                c: &mut CudaSlice<Self>,
            ) -> Result<(), CublasError> {
                // cuBLAS is column-major: C^T = B^T A^T.
                let cfg = GemmConfig {
                    transa: cublasOperation_t::CUBLAS_OP_N,
                    transb: cublasOperation_t::CUBLAS_OP_N,
                    m: n as i32,
                    n: m as i32,
                    k: k as i32,
                    alpha: 1.0 as $t,
                    lda: n as i32,
                    ldb: k as i32,
                    beta: 0.0 as $t,
                    ldc: n as i32,
                };
                unsafe { blas.gemm(cfg, b, a, c) }
            }
        }
    };
}

impl_cuda_scalar!(f32, "float");
impl_cuda_scalar!(f64, "double");
