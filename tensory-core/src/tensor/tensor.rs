use crate::tensor::{TensorRepr, broker::TensorBroker};

pub struct Tensor<B: TensorBroker, T: TensorRepr> {
    repr: T,
    broker: B,
}

impl<B: TensorBroker, T: TensorRepr> Tensor<B, T> {
    // conversion
    pub fn from_raw(repr: T, broker: B) -> Result<Self, (T, B)> {
        if broker.len() == repr.dim() {
            Ok(unsafe { Self::from_raw_unchecked(repr, broker) })
        } else {
            Err((repr, broker))
        }
    }
    pub unsafe fn from_raw_unchecked(repr: T, broker: B) -> Self {
        Self { repr, broker }
    }
    pub fn into_raw(self) -> (T, B) {
        (self.repr, self.broker)
    }

    // accessor
    pub fn broker(&self) -> &B {
        &self.broker
    }
    // these mut fn is obsoluted because breaks the invariant dim == len
    // planed to replace by runtime invariant checker
    /// # Safety
    ///
    /// caller must not swap the object
    pub unsafe fn broker_mut(&mut self) -> &mut B {
        &mut self.broker
    }
    pub fn repr(&self) -> &T {
        &self.repr
    }

    /// # Safety
    ///
    /// caller must not swap the object
    pub unsafe fn repr_mut(&mut self) -> &mut T {
        &mut self.repr
    }
}

// struct TensorMutRefGuard<'a, M:AxisMgr, T:TensorRepr> {
//     raw: &'a mut T,
//     mgr: &'a mut M,
// }

// impl<'a, M: AxisMgr, T: TensorRepr> Drop for TensorMutRefGuard<'a, M, T> {
//     fn drop(&mut self) {
//     self.raw
//         // self.mgr.borrow_mut().use_mut();
//         // self.raw.use_mut();
//         // self.mgr.use_mut();
//     }
// }

// impl<'a, M, T> TensorMutRefGuard<'a, M, T> {
//     f
// }

// pub fn replace_leg(&mut self, old_leg: &LA::Id, new_leg: LA::Id) -> Result<LA::Id, LA::Id> {
//     self.leg_alloc.replace(old_leg, new_leg)
// }

#[cfg(test)]
mod tests {

    use std::println;

    use super::*;

    #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
    struct DummyTensor(usize);
    unsafe impl TensorRepr for DummyTensor {
        fn dim(&self) -> usize {
            self.0
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
    struct DummyLegId;

    #[test]
    fn it_works() {
        let raw_tensor = DummyTensor(1);

        let ts = Tensor::from_raw(raw_tensor, leg_set![DummyLegId]).unwrap();

        println!("{:?}", ts.broker());
    }
}
