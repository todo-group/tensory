//! cudarc backed dense tensor representation, mirroring `tensory-ndarray`.

pub mod arith;
pub mod naive;
mod runtime;
mod scalar;

pub use runtime::{CudaErr, CudaRuntime};
pub use scalar::CudaScalar;

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use tensory_core::{
    args::LegMapArg,
    mapper::{AxisMapper, BuildableMapper},
    repr::{AsViewRepr, IntoOwnedRepr, TensorTupleRepr},
    tensor::Tensor,
    utils::axis_info::AxisInfoReprImpl,
};
use thiserror::Error;

/// Namespace marker for the cuda implementation.
pub struct Cuda;

/// Dense device buffer together with its tensor layout (shape and element strides).
#[derive(Debug)]
pub struct CudaDenseRepr<E> {
    data: CudaSlice<E>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// Immutable view of [`CudaDenseRepr`].
#[derive(Debug)]
pub struct CudaDenseViewRepr<'a, E> {
    data: &'a CudaSlice<E>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<E> Clone for CudaDenseViewRepr<'_, E> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut acc = 1;
    for (st, &s) in strides.iter_mut().zip(shape).rev() {
        *st = acc;
        acc *= s;
    }
    strides
}

pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1;
    for (&s, &st) in shape.iter().zip(strides).rev() {
        if s != 1 && st != expected {
            return false;
        }
        expected *= s;
    }
    true
}

fn to_host<E: CudaScalar>(
    data: &CudaSlice<E>,
    shape: &[usize],
    strides: &[usize],
) -> Result<Vec<E>, CudaErr> {
    let raw = data.stream().clone_dtoh(data)?;
    if is_contiguous(shape, strides) {
        return Ok(raw);
    }
    let len: usize = shape.iter().product();
    let mut out = Vec::with_capacity(len);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..len {
        out.push(raw[idx.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>()]);
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

impl<E> CudaDenseRepr<E> {
    /// Creates a contiguous (row-major) representation from a device buffer.
    pub fn from_raw(data: CudaSlice<E>, shape: Vec<usize>) -> Result<Self, CudaErr> {
        if data.len() != shape.iter().product::<usize>() {
            return Err(CudaErr::InvalidInput);
        }
        let strides = contiguous_strides(&shape);
        Ok(Self {
            data,
            shape,
            strides,
        })
    }
    /// Creates a representation from a buffer and an arbitrary layout; the layout must address exactly the buffer.
    pub(crate) fn from_raw_layout(
        data: CudaSlice<E>,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }
    pub fn into_raw(self) -> (CudaSlice<E>, Vec<usize>, Vec<usize>) {
        (self.data, self.shape, self.strides)
    }
    pub fn data(&self) -> &CudaSlice<E> {
        &self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl<E: CudaScalar> CudaDenseRepr<E> {
    /// Uploads row-major host data.
    pub fn from_host(
        stream: &Arc<CudaStream>,
        shape: Vec<usize>,
        data: &[E],
    ) -> Result<Self, CudaErr> {
        Self::from_raw(stream.clone_htod(data)?, shape)
    }
    pub fn zeros(stream: &Arc<CudaStream>, shape: Vec<usize>) -> Result<Self, CudaErr> {
        let len = shape.iter().product();
        Self::from_raw(stream.alloc_zeros::<E>(len)?, shape)
    }
    /// Downloads the elements in logical row-major order.
    pub fn to_host(&self) -> Result<Vec<E>, CudaErr> {
        to_host(&self.data, &self.shape, &self.strides)
    }
}

impl<'a, E> CudaDenseViewRepr<'a, E> {
    pub fn data(&self) -> &'a CudaSlice<E> {
        self.data
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl<E: CudaScalar> CudaDenseViewRepr<'_, E> {
    /// Downloads the elements in logical row-major order.
    pub fn to_host(&self) -> Result<Vec<E>, CudaErr> {
        to_host(self.data, &self.shape, &self.strides)
    }
}

unsafe impl<E> TensorTupleRepr<1> for CudaDenseRepr<E> {
    fn naxes_array(&self) -> [usize; 1] {
        [self.shape.len()]
    }
}
unsafe impl<E> TensorTupleRepr<1> for CudaDenseViewRepr<'_, E> {
    fn naxes_array(&self) -> [usize; 1] {
        [self.shape.len()]
    }
}

unsafe impl<'a, E: 'a> AsViewRepr<'a, 1> for CudaDenseRepr<E> {
    type View = CudaDenseViewRepr<'a, E>;
    fn view(&'a self) -> Self::View {
        CudaDenseViewRepr {
            data: &self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}
unsafe impl<'a, 'b, E> AsViewRepr<'b, 1> for CudaDenseViewRepr<'a, E> {
    type View = CudaDenseViewRepr<'a, E>;
    fn view(&'b self) -> Self::View {
        self.clone()
    }
}

unsafe impl<E> IntoOwnedRepr<1> for CudaDenseRepr<E> {
    type Owned = CudaDenseRepr<E>;
    fn into_owned(self) -> Self::Owned {
        self
    }
}
unsafe impl<E: CudaScalar> IntoOwnedRepr<1> for CudaDenseViewRepr<'_, E> {
    type Owned = CudaDenseRepr<E>;
    fn into_owned(self) -> Self::Owned {
        // The trait is infallible, so a device copy failure panics.
        let data = self
            .data
            .stream()
            .clone_dtod(self.data)
            .expect("device copy failed");
        CudaDenseRepr::from_raw_layout(data, self.shape, self.strides)
    }
}

impl<E> AxisInfoReprImpl<'_, 1> for CudaDenseRepr<E> {
    type AxisInfo = usize;
    unsafe fn axis_info_unchecked(&self, _i: usize, j: usize) -> Self::AxisInfo {
        self.shape[j]
    }
}
impl<E> AxisInfoReprImpl<'_, 1> for CudaDenseViewRepr<'_, E> {
    type AxisInfo = usize;
    unsafe fn axis_info_unchecked(&self, _i: usize, j: usize) -> Self::AxisInfo {
        self.shape[j]
    }
}

pub type CudaDenseTensor<E, B> = Tensor<CudaDenseRepr<E>, B>;
pub type CudaDenseViewTensor<'a, E, B> = Tensor<CudaDenseViewRepr<'a, E>, B>;

#[derive(Debug, Error)]
pub enum CudaDenseFromHostErr<M> {
    #[error("Cuda operation failed: {0}")]
    Cuda(#[from] CudaErr),
    #[error("Mapper build failed: {0}")]
    MapperBuild(M),
}

pub trait CudaDenseTensorExt<E, M: AxisMapper>: Sized {
    /// Uploads row-major host data as a tensor.
    fn from_host<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        stream: &Arc<CudaStream>,
        map: LegMapArg<K, V>,
        data: &[E],
    ) -> Result<Self, CudaDenseFromHostErr<<M as BuildableMapper<K>>::Err>>
    where
        M: BuildableMapper<K>;
}

impl<E: CudaScalar, M: AxisMapper> CudaDenseTensorExt<E, M> for CudaDenseTensor<E, M> {
    fn from_host<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        stream: &Arc<CudaStream>,
        map: LegMapArg<K, V>,
        data: &[E],
    ) -> Result<Self, CudaDenseFromHostErr<<M as BuildableMapper<K>>::Err>>
    where
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::build(k).map_err(CudaDenseFromHostErr::MapperBuild)?;
        let repr = CudaDenseRepr::from_host(stream, v.collect(), data)?;
        Ok(unsafe { Tensor::from_raw_unchecked(repr, [mapper]) })
    }
}
