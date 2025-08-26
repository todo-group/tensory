#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod svd;

pub mod qr;

pub mod lu;

pub mod eigen;

// you can add your own traits as we defined!

// pub unsafe etrait QrContextImpl
// pub unsafe trait LuContextImpl
// pub unsafe trait CholeskyContextImpl
// pub unsafe trait EigenContextImpl
// pub unsafe trait ExpContextImpl
