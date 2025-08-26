use core::{error::Error, fmt::Display};
use ndarray_linalg::error::LinalgError;

#[derive(Debug)]
pub enum TenalgError {
    InvalidInput,
    InvalidResult,
    Linalg(LinalgError),
}

impl Display for TenalgError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            TenalgError::InvalidInput => {
                write!(f, "Tenalg received invalid input")
            }
            TenalgError::InvalidResult => {
                write!(f, "Linalg returned invalid result")
            }
            TenalgError::Linalg(le) => {
                write!(f, "Linalg failed: {}", le)
            }
        }
    }
}

impl Error for TenalgError {}

impl<T> From<T> for TenalgError
where
    LinalgError: From<T>,
{
    fn from(source: T) -> Self {
        TenalgError::Linalg(source.into())
    }
}
