#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

/// define basic concepts of tensor
pub mod tensor;

/// provide fundumental operations for tensor
pub mod ops;

/// provide elemental arithmetics operations, with operator overloading
pub mod arith;
