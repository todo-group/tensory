//! Bike-shed modules for future implementations. These modules are intended to be used as a placeholder for future implementations, and are not intended to be used in the current version of the library. They are not intended to be used in the current version of the library, and are not intended to be used in the current version of the library.

use thiserror::Error;

/// bike-shed error type.
#[derive(Error, Debug, Clone, PartialEq, Eq, Hash)]
#[error("Port error")]
pub struct PortError;
