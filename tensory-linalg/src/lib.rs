#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod svd;

pub mod qr;

pub mod lu;

pub mod eig;

pub mod det;

pub mod norm;

pub mod conj;

pub mod exp;

pub mod pow;

pub mod solve_eig;

// pub mod lu;

// pub mod cholesky;

pub mod prelude {
    pub use crate::conj::TensorConjExt;
    // pub use crate::det::TensorDetExt;
    pub use crate::eig::TensorEigExt;
    // pub use crate::exp::TensorExpExt;
    // pub use crate::lu::TensorLuExt;
    pub use crate::norm::TensorNormExt;
    // pub use crate::pow::TensorPowExt;
    pub use crate::qr::TensorQrExt;
    pub use crate::solve_eig::TensorSolveEigExt;
    pub use crate::svd::TensorSvdExt;

    // pub use crate::conj::BoundTensorConjExt;
    // pub use crate::det::BoundTensorDetExt;
    // pub use crate::eig::BoundTensorEigExt;
    // pub use crate::exp::BoundTensorExpExt;
    // pub use crate::lu::BoundTensorLuExt;
    pub use crate::norm::BoundTensorNormExt;
    // pub use crate::pow::BoundTensorPowExt;
    pub use crate::qr::BoundTensorQrExt;
    // pub use crate::solve_eig::BoundTensorSolveEigExt;
    pub use crate::svd::BoundTensorSvdExt;
}
