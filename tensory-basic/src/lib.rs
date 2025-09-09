#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

/// provide itensor-like Id
pub mod id;

/// provide referencial TensorBroker implementations
pub mod broker;
