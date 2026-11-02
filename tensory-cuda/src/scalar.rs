use std::ops::Neg;

use cudarc::{
    cutensor::sys::{cudaDataType_t, cutensorComputeDescriptor_t},
    driver::{DeviceRepr, ValidAsZeroBits},
};

// cuTENSOR exports its compute descriptors as data symbols, which cudarc does not expose.
#[link(name = "cutensor")]
unsafe extern "C" {
    static CUTENSOR_COMPUTE_DESC_32F: cutensorComputeDescriptor_t;
    static CUTENSOR_COMPUTE_DESC_64F: cutensorComputeDescriptor_t;
}

/// Element types supported by the cuda implementation.
pub trait CudaScalar:
    DeviceRepr + ValidAsZeroBits + Copy + Unpin + Neg<Output = Self> + 'static
{
    const DATA_TYPE: cudaDataType_t;
    const ONE: Self;
    const ZERO: Self;
    fn compute_desc() -> cutensorComputeDescriptor_t;
}

impl CudaScalar for f32 {
    const DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_32F;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { CUTENSOR_COMPUTE_DESC_32F }
    }
}

impl CudaScalar for f64 {
    const DATA_TYPE: cudaDataType_t = cudaDataType_t::CUDA_R_64F;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    fn compute_desc() -> cutensorComputeDescriptor_t {
        unsafe { CUTENSOR_COMPUTE_DESC_64F }
    }
}
