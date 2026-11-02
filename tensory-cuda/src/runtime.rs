use std::{rc::Rc, sync::Arc};

use cudarc::{
    cutensor::result::CutensorError,
    driver::{CudaContext, CudaStream, DriverError},
};
use tensory_core::concept::task::IsRuntime;
use thiserror::Error;

use crate::cutensor::Handle;

#[derive(Debug, Error)]
pub enum CudaErr {
    #[error("Invalid input")]
    InvalidInput,
    #[error("Driver error: {0}")]
    Driver(#[from] DriverError),
    #[error("cuTENSOR error: {0}")]
    Cutensor(#[from] CutensorError),
}

/// Context for executing tasks: a stream and a cuTENSOR handle bound to it.
#[derive(Clone)]
pub struct CudaRuntime {
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) handle: Rc<Handle>,
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
        stream.context().bind_to_thread()?;
        Ok(Self {
            stream,
            handle: Rc::new(Handle::new()?),
        })
    }
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
