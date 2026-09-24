//! Core crate of tensory.

#![warn(missing_docs)]
#![allow(clippy::type_complexity)]
#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

// utilities for general-purpose operations and helpers

pub mod concept;

pub mod bikeshed;

// core concepts

pub mod repr;

pub mod mapper;

pub mod tensor;

// functionalitys built on core concepts
// they are in the core crate due to one or more reason below:
// - they are fundamental enough to be in the core crate
// - they require std trait implementations (e.g. Add, Mul, etc) which are not allowed in other crates

pub mod op;

pub mod mem;

pub mod arith;

pub mod utils;

// common

pub mod args;

// mod veccy;

pub mod prelude {
    //! A prelude module re-exporting commonly used items.

    pub use crate::concept::container::*;
    pub use crate::concept::task::*;
    pub use crate::mem::*;

    //pub use crate::bound_tensor::*;
    pub use crate::mapper::*;
    pub use crate::repr::*;
    pub use crate::tensor::{Tensor, TensorExt, TensorTuple, ToTensor, ToTensorTuple};

    pub use crate::leg;
    pub use crate::lm;
    pub use crate::ls;
}
