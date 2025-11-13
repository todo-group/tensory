//! Provide pluggable way to implement arithmetic operations.
//!
//! Potentially, we could suupport all the arithmetic operations defined in `core` crate, listed below (`a` and `b` are tensors, `#` are non-tensors):
//!
//! - numeric unary: `- a`
//! - numeric binary: `a + b`, `a + #` (?), `# + a` (?), `a - b`, `a - #` (?), `# - a` (?), `a * b` (!), `a * #`, `# * a`, `a / b` (?), `a / #`, `# / b` (?), `a % b` (?), `a % #`, `# % a` (?)
//! - bitwise unary: `! a`
//! - bitwise binary: `a & b` (?), `a & #`, `# & a`,  `a | b` (?), `a | #`, `# | a`, `a ^ b` (?), `a ^ #`, `# ^ a` (?)
//! - shift: `a >> b` (?), `a >> #` (?), `# >> a` (?), `a << b` (?), `a << E` (?), `E << a` (?)
//!
//! Now we only support the following operations listed below:
//!
//! - `- a`
//! - `a + b`
//! - `a - b`
//! - `a * b` (!)
//! - `a * #`
//! - `# * a` (may be removed)
//! - `a / #`
//! - `a ^ #`
//!
//! these are roughly covers the operations dinstincted in (non-comutative) group, ring, and field. (left subtraction and division are missed, but they are expected tp be same as `#-a` := `-(a-#)`, `#/a` := `1/# * a`)
//!

/// provide interface for tensor-tensor addition
mod add;
pub use add::*;

/// provide interface for tensor-tensor subtraction
mod sub;
pub use sub::*;

/// provide interface for tensor negation.
mod neg;
pub use neg::*;

/// provide interface for tensor-scalar multiplication
mod scalar_mul;
pub use scalar_mul::*;

/// provide interface for tensor-scalar division
mod scalar_div;
pub use scalar_div::*;

// /// provide interface for tensor-conjugation
// mod conj;
// pub use conj::*;

/// provide interface for tensor-tensor multiplication (contraction)
mod contr;
pub use contr::*;

/// A marker trait for types that can be used as scalars in tensor arithmetic operations.
pub trait TensorScalar {}
impl TensorScalar for bool {}
impl TensorScalar for i8 {}
impl TensorScalar for i16 {}
impl TensorScalar for i32 {}
impl TensorScalar for i64 {}
impl TensorScalar for i128 {}
impl TensorScalar for isize {}
impl TensorScalar for u8 {}
impl TensorScalar for u16 {}
impl TensorScalar for u32 {}
impl TensorScalar for u64 {}
impl TensorScalar for u128 {}
impl TensorScalar for usize {}
//impl TensorScalar for f16 {}
impl TensorScalar for f32 {}
impl TensorScalar for f64 {}
// impl TensorScalar for Complex<f32> {}
// impl TensorScalar for Complex<f64> {}

//impl TensorScalar for f128 {}
