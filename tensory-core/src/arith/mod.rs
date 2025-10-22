//! Provide pluggable way to implement arithmetic operations.
//!
//! Potentially, we could suupport all the arithmetic operations defined in `core` crate:
//!
//!
//! Here we list the operations. `a`,`b` are tensors, `#` are non-tensors.
//!
//! numeric unary:
//!
//! - `- a`
//!
//! numeric binary:
//!
//! - `a + b`
//! - `a + #` (?)
//! - `# + a` (?)
//! - `a - b`
//! - `a - #` (?)
//! - `# - a` (?)
//! - `a * b` (! special !)
//! - `a * #`
//! - `# * a`
//! - `a / b` (?)
//! - `a / #`
//! - `# / b` (?)
//! - `a % b` (?)
//! - `a % #`
//! - `# % a` (?)
//!
//! bitwise unary:
//!
//! - `! a`
//!
//! bitwise binary:
//!
//! - `a & b` (?)
//! - `a & #`
//! - `# & a`
//!
//! - `a | b` (?)
//! - `a | #`
//! - `# | a`
//!
//! - `a ^ b` (?)
//! - `a ^ #`
//! - `# ^ a` (?)
//!
//! shift:
//!
//! - `a >> b` (?)
//! - `a >> #` (?)
//! - `# >> a` (?)
//! - `a << b` (?)
//! - `a << E` (?)
//! - `E << a` (?)
//!
//!
//! Now we only support the following operations:
//!
//!
//! - `- a`
//! - `a + b`
//! - `a - b`
//! - `a * b`
//! - `a * #`
//! - `# * a`
//! - `a / #`
//! - `a ^ #`
//!
//! these are roughly covers the operations dinstincted in (non-comutative) group, ring, and field. (left subtraction and division are missed, but they are expected tp be same as `#-a` := `-(a-#)`, `#/a` := `1/# * a`)
//!

// macro_rules! impl_binary {
//     ($l:ty,$r:ty $(,$life:lifetime)*) => {
//         impl<$($life,)* L: TensorRepr, R: TensorRepr, M: ConnectMapper<2>, RT> Mul<$r> for $l
//         where
//             $l: ToBoundTensor<Mapper = M, Runtime = RT>,
//             $r: ToBoundTensor<Mapper = M, Runtime = RT>,
//             RT: Copy + Eq + MulRuntime<<$l as ToBoundTensor>::Repr, <$r as ToBoundTensor>::Repr>,
//         {
//             type Output = Result<
//                 BoundTensor<
//                     <<RT as MulRuntime<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Ctx as MulCtxImpl<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Res,
//                     M,
//                     RT,
//                 >,
//                 RuntimeError<
//                     <M as ConnectMapper<2>>::Err,
//                     <<RT as MulRuntime<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Ctx as MulCtxImpl<
//                         <$l as ToBoundTensor>::Repr,
//                         <$r as ToBoundTensor>::Repr,
//                     >>::Err,
//                 >,
//             >;
//             fn mul(self, rhs: $r) -> Self::Output {
//                 let (lhs, lhs_rt) = self.to_bound_tensor().into_raw();
//                 let (rhs, rhs_rt) = rhs.to_bound_tensor().into_raw();

//                 if lhs_rt != rhs_rt {
//                     return Err(RuntimeError::Runtime);
//                 }
//                 let res = (lhs * rhs)
//                     .map_err(RuntimeError::Axis)?
//                     .with(lhs_rt.mul_ctx())
//                     .map_err(RuntimeError::Ctx)?;
//                 Ok(BoundTensor::from_raw(res, lhs_rt))
//             }
//         }
//     };
// }

/// provide interface for tensor-tensor addition
mod add;
pub use add::*;

// /// provide interface for tensor-tensor subtraction
// mod sub;
// pub use sub::*;

// /// provide interface for tensor additive inverse, and the induced subtractions
// mod neg;
// pub use neg::*;

/// provide interface for tensor-scalar multiplication
mod scalar_mul;
pub use scalar_mul::*;

/// provide interface for tensor-scalar division
mod scalar_div;
pub use scalar_div::*;

/// provide interface for tensor-conjugation
mod conj;
pub use conj::*;

/// provide interface for tensor-tensor multiplication (contraction)
mod contr;
pub use contr::*;
