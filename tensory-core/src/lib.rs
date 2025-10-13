//! Core crate of tensory.

#![warn(missing_docs)]
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

// core concepts

pub mod repr;

pub mod mapper;

pub mod tensor;

pub mod tensor_with_runtime;

// functionalitys built on core concepts
// they are in the core crate due to one or more reason below:
// - they are fundamental enough to be in the core crate
// - they require std trait implementations (e.g. Add, Mul, etc) which are not allowed in other crates

pub mod arith;

pub mod utils;

// common

pub mod args;
