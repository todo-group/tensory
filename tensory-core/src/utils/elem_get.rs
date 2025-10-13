use alloc::vec::Vec;

use crate::{
    args::LegMapArg,
    mapper::{AxisMapper, TranslateMapper},
    repr::TensorRepr,
    tensor::Tensor,
};

/// Tensor representation providing immutable element access, WITHOUT checking the number of indices.
pub trait ElemGetReprImpl: TensorRepr {
    type Index;
    type E;
    type Err;
    /// Returns the immutable reference to the element at the given indices, WITHOUT checking the number of indices.
    unsafe fn get_unchecked(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err>;
}

/// Tensor representation providing mutable element access, WITHOUT checking the number of indices.
pub trait ElemGetMutReprImpl: ElemGetReprImpl {
    /// Returns the mutable reference to the element at the given indices, WITHOUT checking the number of indices.
    unsafe fn get_mut_unchecked(
        &mut self,
        indices: Vec<Self::Index>,
    ) -> Result<&mut Self::E, Self::Err>;
}

/// Safe version of `ElementAccessImpl`.
///
/// The blanket implementation checks the number of indices.
pub trait ElemGetRepr: ElemGetReprImpl {
    fn get(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err>;
}
impl<T: ElemGetReprImpl> ElemGetRepr for T {
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
pub trait ElemGetMutRepr: ElemGetMutReprImpl {
    fn get_mut(&mut self, indices: Vec<Self::Index>) -> Result<&mut Self::E, Self::Err>;
}
impl<T: ElemGetMutReprImpl> ElemGetMutRepr for T {
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

impl<A: ElemGetRepr, B: AxisMapper> Tensor<A, B> {
    pub fn get<
        'a,
        K: ExactSizeIterator + Iterator<Item = &'a B::Id>,
        V: ExactSizeIterator + Iterator<Item = A::Index>,
    >(
        &self,
        map: LegMapArg<K, V>,
    ) -> Result<Result<&A::E, A::Err>, B::Err>
    where
        B: TranslateMapper<A::Index>,
        B::Id: 'a,
    {
        let v = self.mapper().translate(map)?;
        Ok(unsafe { self.repr().get_unchecked(v) })
    }
}
impl<A: ElemGetMutRepr, B: AxisMapper> Tensor<A, B> {
    pub fn get_mut<
        'a,
        K: ExactSizeIterator + Iterator<Item = &'a B::Id>,
        V: ExactSizeIterator + Iterator<Item = A::Index>,
    >(
        &mut self,
        map: LegMapArg<K, V>,
    ) -> Result<Result<&mut A::E, A::Err>, B::Err>
    where
        B: TranslateMapper<A::Index>,
        B::Id: 'a,
    {
        let v = { self.mapper().translate(map)? };
        Ok(unsafe { self.repr_mut().get_mut_unchecked(v) })
    }
}

// impl<M: TensorBroker, T: ElemGetMut> Tensor<M, T> {
//     pub fn get_mut<Map>(&mut self, indices: Map) -> Result<Result<&mut T::E, T::Err>, M::Err>
//     where
//         M: TranslateBroker<Map, Content = T::Index>,
//     {
//         let v = self.broker().translate(indices)?;
//         Ok(unsafe { self.repr_mut().get_mut_unchecked(v) })
//     }
// }

impl<
    'a,
    A: ElemGetRepr,
    B: TranslateMapper<A::Index>,
    K: ExactSizeIterator + Iterator<Item = &'a B::Id>,
    V: ExactSizeIterator + Iterator<Item = A::Index>,
> core::ops::Index<LegMapArg<K, V>> for Tensor<A, B>
where
    A::Err: core::fmt::Debug,
    B::Err: core::fmt::Debug,
    B::Id: 'a,
{
    type Output = A::E;

    fn index(&self, indices: LegMapArg<K, V>) -> &Self::Output {
        self.get(indices).unwrap().unwrap()
    }
}
impl<
    'a,
    T: ElemGetMutRepr,
    B: TranslateMapper<T::Index>,
    K: ExactSizeIterator + Iterator<Item = &'a B::Id>,
    V: ExactSizeIterator + Iterator<Item = T::Index>,
> core::ops::IndexMut<LegMapArg<K, V>> for Tensor<T, B>
where
    T::Err: core::fmt::Debug,
    B::Err: core::fmt::Debug,
    B::Id: 'a,
{
    fn index_mut(&mut self, indices: LegMapArg<K, V>) -> &mut Self::Output {
        self.get_mut(indices).unwrap().unwrap()
    }
}
