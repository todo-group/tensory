use std::collections::HashMap;
use std::ops::Mul;

use bimap::BiHashMap;

use crate::core::tensor_repr::{ContractionContext, ContractionIndexProvenance};

use super::tensor_repr::{ElementAccess, TensorRepr};

use std::fmt::Debug;
use std::hash::Hash;

pub trait LegId: Debug + Eq + Ord + Hash + Clone {
    //type IV: LegIndexRepr<Self>;
    //fn dim(&self) -> usize;
    //fn fix(&self, loc: &Self::L) -> Self::IV;
}
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LegVal<Id: LegId, I>(pub Id, pub I);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LegRefVal<'a, Id: LegId, I>(pub &'a Id, pub I);

#[macro_export]
macro_rules! v {
    ( $( $x:expr => $y:expr),* ) => {
        [$($crate::core::tensor::LegVal($x.clone(), $y)),*]
    };
}

#[macro_export]
macro_rules! rv {
    ( $( $x:expr => $y:expr),* ) => {
        [$($crate::core::tensor::LegRefVal(&$x, $y)),*]
    };
}

pub struct Tensor<Id: LegId, T: TensorRepr> {
    raw: T,
    // Id should unique
    legs: BiHashMap<usize, Id>,
}

impl<Id: LegId, T: TensorRepr> Tensor<Id, T> {
    pub fn from_raw<I: IntoIterator<Item = Id>>(raw: T, legs: I) -> Result<Self, T> {
        let legs: BiHashMap<usize, Id> = legs.into_iter().enumerate().collect();
        if legs.len() == raw.len() {
            Ok(Self { raw, legs })
        } else {
            Err(raw)
        }
    }
    pub fn from_raw_and_bimap(raw: T, legs: BiHashMap<usize, Id>) -> Result<Self, T> {
        if legs.len() == raw.len() {
            Ok(Self { raw, legs })
        } else {
            Err(raw)
        }
    }
    pub fn legs(&self) -> &BiHashMap<usize, Id> {
        &self.legs
    }
    pub fn replace_leg(&mut self, old_leg: &Id, new_leg: Id) -> Result<Id, Id> {
        if self.legs.contains_right(&new_leg) {
            return Err(new_leg);
        }
        match self.legs.remove_by_right(old_leg) {
            None => return Err(new_leg),
            Some((idx, id)) => {
                self.legs.insert(idx, new_leg);
                return Ok(id);
            }
        }
    }
    pub fn raw(&self) -> &T {
        &self.raw
    }
}

pub struct TensorMul<Id: LegId, L: TensorRepr, R: TensorRepr> {
    lhs: Tensor<Id, L>,
    rhs: Tensor<Id, R>,
}

impl<Id: LegId, L: TensorRepr, R: TensorRepr> TensorMul<Id, L, R> {
    pub fn new(lhs: Tensor<Id, L>, rhs: Tensor<Id, R>) -> Self {
        Self { lhs, rhs }
    }
    pub fn by<C: ContractionContext<L, R>>(self, context: C) -> Result<Tensor<Id, C::Res>, C::Err> {
        let lhs_legs = self.lhs.legs;
        let rhs_legs = self.rhs.legs;
        let lhs_raw = self.lhs.raw;
        let rhs_raw = self.rhs.raw;

        let idx_pairs: Vec<(usize, usize)> = lhs_legs
            .iter()
            .filter_map(|(lhs_idx, id)| {
                rhs_legs
                    .get_by_right(id)
                    .map(|rhs_idx| (*lhs_idx, *rhs_idx))
            })
            .collect();

        let (res, idx_trans) = context.contract(lhs_raw, rhs_raw, &idx_pairs)?;

        let mut lhs_legs = lhs_legs;
        let mut rhs_legs = rhs_legs;

        let legs: BiHashMap<usize, Id> = idx_trans
            .into_iter()
            .enumerate()
            .map(|(new_idx, (prov, idx))| {
                (
                    new_idx,
                    match prov {
                        ContractionIndexProvenance::Lhs => lhs_legs.remove_by_left(&idx).unwrap().1,
                        ContractionIndexProvenance::Rhs => rhs_legs.remove_by_left(&idx).unwrap().1,
                    },
                )
            })
            .collect();
        Ok(Tensor { raw: res, legs })
    }
}

impl<Id: LegId, L: TensorRepr, R: TensorRepr> Mul<Tensor<Id, R>> for Tensor<Id, L> {
    type Output = TensorMul<Id, L, R>;

    fn mul(self, rhs: Tensor<Id, R>) -> Self::Output {
        TensorMul::new(self, rhs)
    }
}

impl<Id: LegId, T: ElementAccess> Tensor<Id, T> {
    pub fn get<'a>(
        &self,
        index: impl IntoIterator<Item = LegRefVal<'a, Id, T::Index>>,
    ) -> Result<&T::E, T::Err>
    where
        Id: 'a,
    {
        let mut index_set: HashMap<_, _> = index
            .into_iter()
            .map(|LegRefVal(id, idx)| (id, idx))
            .collect();

        let idxs: Vec<T::Index> = (0..self.legs.len())
            .into_iter()
            .map(|i| {
                index_set
                    .remove(self.legs.get_by_left(&i).unwrap())
                    .unwrap()
            })
            .collect();
        self.raw.get(&idxs)
    }
    pub fn get_mut<'a>(
        &mut self,
        index: impl IntoIterator<Item = LegRefVal<'a, Id, T::Index>>,
    ) -> Result<&mut T::E, T::Err>
    where
        Id: 'a,
    {
        let mut index_set: HashMap<_, _> = index
            .into_iter()
            .map(|LegRefVal(id, idx)| (id, idx))
            .collect();

        let idxs: Vec<T::Index> = (0..self.legs.len())
            .into_iter()
            .map(|i| {
                index_set
                    .remove(self.legs.get_by_left(&i).unwrap())
                    .unwrap()
            })
            .collect();
        self.raw.get_mut(&idxs)
    }
}

impl<'a, Id: LegId + 'a, T: ElementAccess, I: IntoIterator<Item = LegRefVal<'a, Id, T::Index>>>
    std::ops::Index<I> for Tensor<Id, T>
{
    type Output = T::E;

    fn index(&self, index: I) -> &Self::Output {
        self.get(index).unwrap()
    }
}
impl<'a, Id: LegId + 'a, T: ElementAccess, I: IntoIterator<Item = LegRefVal<'a, Id, T::Index>>>
    std::ops::IndexMut<I> for Tensor<Id, T>
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
    struct DummyTensor(usize);
    impl TensorRepr for DummyTensor {
        fn len(&self) -> usize {
            self.0
        }
    }

    #[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
    struct DummyLegId;
    impl LegId for DummyLegId {}

    #[test]
    fn it_works() {
        let raw_tensor = DummyTensor(1);

        let ts = Tensor::from_raw(raw_tensor, vec![DummyLegId]).unwrap();

        println!("{:?}", ts.legs());
    }
}
