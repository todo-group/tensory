/// core traits and structs low and middle layer implementation
pub mod core;

/// basical implementors
pub mod basic;

/// numerical dense tensor implemented with ndarray
/// note: since ndarray is NOT aware of memory allocation and dynamic library call, this module will be replaced in the future by module `dense`, which is planned to call BLAS or other linear algebra libraries more directly.
pub mod nd_dense;
