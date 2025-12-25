use core::fmt::Display;
use ndarray_linalg::error::LinalgError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TenalgErr {
    #[error("Tenalg received invalid input")]
    InvalidInput,
    #[error("Linalg returned invalid result")]
    InvalidResult,
    #[error("Linalg failed: {0}")]
    Linalg(LinalgError),
}

impl<T> From<T> for TenalgErr
where
    LinalgError: From<T>,
{
    fn from(source: T) -> Self {
        TenalgErr::Linalg(source.into())
    }
}
