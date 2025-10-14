#![warn(missing_docs)]
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

/// provide itensor-like Id
pub mod id;

/// provide referencial AxisMapper implementations
pub mod mapper;
