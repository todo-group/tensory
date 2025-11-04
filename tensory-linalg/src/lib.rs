#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod svd;

pub mod qr;

pub mod lu;

pub mod eig;

pub mod norm;

pub mod conj;

// you can add your own traits as we defined!

// pub unsafe trait LuContextImpl
// pub unsafe trait CholeskyContextImpl
// pub unsafe trait ExpContextImpl
