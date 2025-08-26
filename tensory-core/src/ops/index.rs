use alloc::vec::Vec;

use crate::tensor::{Tensor, TensorBroker, TensorRepr, TranslateMgr};

/// Tensor representation providing immutable element access, WITHOUT checking the number of indices.
pub trait ElemGetImpl: TensorRepr {
    type Index;
    type E;
    type Err;
    /// Returns the immutable reference to the element at the given indices, WITHOUT checking the number of indices.
    unsafe fn get_unchecked(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err>;
}

/// Tensor representation providing mutable element access, WITHOUT checking the number of indices.
pub trait ElemGetMutImpl: ElemGetImpl {
    /// Returns the mutable reference to the element at the given indices, WITHOUT checking the number of indices.
    unsafe fn get_mut_unchecked(
        &mut self,
        indices: Vec<Self::Index>,
    ) -> Result<&mut Self::E, Self::Err>;
}

/// Safe version of `ElementAccessImpl`.
///
/// The blanket implementation checks the number of indices.
pub trait ElemGet: ElemGetImpl {
    fn get(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err>;
}
impl<T: ElemGetImpl> ElemGet for T {
    fn get(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err> {
        // TODO: check number of indices
        // if number of indices invalid {
        //     panic!();
        // }
        unsafe { self.get_unchecked(indices) }
    }
}

/// Safe version of `ElementAccessImpl`.
///
/// The blanket implementation checks the number of indices.
pub trait ElemGetMut: ElemGetMutImpl {
    fn get_mut(&mut self, indices: Vec<Self::Index>) -> Result<&mut Self::E, Self::Err>;
}
impl<T: ElemGetMutImpl> ElemGetMut for T {
    fn get_mut(&mut self, indices: Vec<Self::Index>) -> Result<&mut Self::E, Self::Err> {
        // TODO: check number of indices
        // if number of indices invalid {
        //     panic!();
        // }
        unsafe { self.get_mut_unchecked(indices) }
    }
}

// #[derive(Debug)]
// pub enum ElementAccessError<Err> {
//     Leg(LegError),
//     Internal(Err),
// }

// impl<Err> core::fmt::Display for ElementAccessError<Err> {
//     fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
//         write!(f, "element access error")
//     }
// }

// impl<Err: core::fmt::Debug> core::error::Error for ElementAccessError<Err> {
//     fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
//         None
//     }
// }

impl<M: TensorBroker, T: ElemGet> Tensor<M, T> {
    pub fn get<Map>(&self, indices: Map) -> Result<Result<&T::E, T::Err>, M::Err>
    where
        M: TranslateMgr<Map, Content = T::Index>,
    {
        let v = self.broker().translate(indices)?;
        Ok(unsafe { self.repr().get_unchecked(v) })
    }
}
impl<M: TensorBroker, T: ElemGetMut> Tensor<M, T> {
    pub fn get_mut<Map>(&mut self, indices: Map) -> Result<Result<&mut T::E, T::Err>, M::Err>
    where
        M: TranslateMgr<Map, Content = T::Index>,
    {
        let v = self.broker().translate(indices)?;
        Ok(unsafe { self.repr_mut().get_mut_unchecked(v) })
    }
}

impl<Map, M: TranslateMgr<Map, Content = T::Index>, T: ElemGet> core::ops::Index<Map>
    for Tensor<M, T>
where
    T::Err: core::fmt::Debug,
    M::Err: core::fmt::Debug,
{
    type Output = T::E;

    fn index(&self, indices: Map) -> &Self::Output {
        self.get(indices).unwrap().unwrap()
    }
}
impl<Map, M: TranslateMgr<Map, Content = T::Index>, T: ElemGetMut> core::ops::IndexMut<Map>
    for Tensor<M, T>
where
    T::Err: core::fmt::Debug,
    M::Err: core::fmt::Debug,
{
    fn index_mut(&mut self, indices: Map) -> &mut Self::Output {
        self.get_mut(indices).unwrap().unwrap()
    }
}
