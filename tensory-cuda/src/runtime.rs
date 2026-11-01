use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use cudarc::{
    cublas::{CudaBlas, result::CublasError},
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    },
    nvrtc::{CompileError, compile_ptx},
};
use tensory_core::concept::task::IsRuntime;
use thiserror::Error;

use crate::CudaScalar;

#[derive(Debug, Error)]
pub enum CudaErr {
    #[error("Invalid input")]
    InvalidInput,
    #[error("Driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("cuBLAS error: {0}")]
    Cublas(#[from] CublasError),
    #[error("NVRTC error: {0}")]
    Nvrtc(#[from] CompileError),
}

/// Runtime owning a stream, a cuBLAS handle and compiled kernels.
#[derive(Clone)]
pub struct CudaRuntime {
    stream: Arc<CudaStream>,
    pub(crate) blas: Arc<CudaBlas>,
    kernels: Arc<Mutex<HashMap<&'static str, CudaFunction>>>,
}

impl PartialEq for CudaRuntime {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.stream, &other.stream)
    }
}
impl Eq for CudaRuntime {}
unsafe impl IsRuntime for CudaRuntime {}

impl CudaRuntime {
    /// Creates a runtime on the default stream of the given device.
    pub fn new(ordinal: usize) -> Result<Self, CudaErr> {
        Self::from_stream(CudaContext::new(ordinal)?.default_stream())
    }
    pub fn from_stream(stream: Arc<CudaStream>) -> Result<Self, CudaErr> {
        Ok(Self {
            blas: Arc::new(CudaBlas::new(stream.clone())?),
            stream,
            kernels: Default::default(),
        })
    }
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Copies a strided buffer into a new contiguous buffer in logical row-major order.
    pub(crate) fn gather<E: CudaScalar>(
        &self,
        src: &CudaSlice<E>,
        shape: &[usize],
        strides: &[usize],
    ) -> Result<CudaSlice<E>, CudaErr> {
        let n: usize = shape.iter().product();
        let mut dst = self.stream.alloc_zeros::<E>(n)?;
        if n == 0 {
            return Ok(dst);
        }

        let func = self.gather_kernel::<E>()?;
        // Device buffers must be non-empty even for 0-dim tensors.
        let to_dev = |v: &[usize]| {
            let mut w: Vec<u64> = v.iter().map(|&x| x as u64).collect();
            w.push(0);
            self.stream.clone_htod(&w)
        };
        let shape_dev = to_dev(shape)?;
        let strides_dev = to_dev(strides)?;

        const BLOCK: u32 = 256;
        let cfg = LaunchConfig {
            block_dim: (BLOCK, 1, 1),
            grid_dim: ((n as u32).div_ceil(BLOCK), 1, 1),
            shared_mem_bytes: 0,
        };
        let (ndim, n) = (shape.len() as i32, n as u64);
        let mut b = self.stream.launch_builder(&func);
        b.arg(src)
            .arg(&mut dst)
            .arg(&shape_dev)
            .arg(&strides_dev)
            .arg(&ndim)
            .arg(&n);
        unsafe { b.launch(cfg) }?;
        Ok(dst)
    }

    fn gather_kernel<E: CudaScalar>(&self) -> Result<CudaFunction, CudaErr> {
        let mut kernels = self.kernels.lock().unwrap();
        if let Some(f) = kernels.get(E::C_NAME) {
            return Ok(f.clone());
        }
        let src = format!(
            r#"
extern "C" __global__ void gather(const {t}* src, {t}* dst, const unsigned long long* shape,
                                  const unsigned long long* strides, int ndim, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long rem = i, off = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        off += (rem % shape[d]) * strides[d];
        rem /= shape[d];
    }}
    dst[i] = src[off];
}}
"#,
            t = E::C_NAME
        );
        let func = self
            .stream
            .context()
            .load_module(compile_ptx(src)?)?
            .load_function("gather")?;
        kernels.insert(E::C_NAME, func.clone());
        Ok(func)
    }
}
