#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

/// provide AxisMgr with "Leg" concept: Axis identified by object.
pub mod id;

pub mod broker;
