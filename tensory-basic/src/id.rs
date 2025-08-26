use core::{fmt, ops::Deref};

use alloc::string::String;
use uuid::Uuid;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Id128 {
    id: Uuid,
}
impl fmt::Display for Id128 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}
impl Default for Id128 {
    fn default() -> Self {
        Self::new()
    }
}

impl Id128 {
    pub fn new() -> Self {
        Self { id: Uuid::new_v4() }
    }
    // fn new_with_prime(dim: usize, plv: usize) -> Self {
    //     Self {
    //         id: Uuid::new_v4(),
    //         plv: plv,
    //     }
    // }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Tag {
    raw: String,
}
impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

//impl LegId for &'static str {}

impl Tag {
    pub fn from_raw(raw: String) -> Self {
        Self { raw }
    }
    // fn new_with_prime(dim: usize, plv: usize) -> Self {
    //     Self {
    //         id: Uuid::new_v4(),
    //         plv: plv,
    //     }
    // }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Prime<Id> {
    id: Id,
    plv: usize,
}
impl<Id> Prime<Id> {
    pub fn new() -> Self
    where
        Id: Default,
    {
        Self {
            id: Id::default(),
            plv: 0,
        }
    }
    pub fn from(id: Id) -> Self {
        Self { id, plv: 0 }
    }
    pub fn prime(self) -> Self {
        self.prime_by(1)
    }
    pub fn deprime(self) -> Self {
        self.deprime_by(1)
    }
    pub fn prime_by(mut self, dplv: usize) -> Self {
        self.plv = self.plv.saturating_add(dplv);
        self
    }
    pub fn deprime_by(mut self, dplv: usize) -> Self {
        self.plv = self.plv.saturating_sub(dplv);
        self
    }
    pub fn plv(&self) -> usize {
        self.plv
    }
}
impl<Id> Deref for Prime<Id> {
    type Target = Id;
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

#[cfg(test)]
mod tests {

    use std::println;

    use super::*;

    type Leg = Prime<Id128>;

    #[test]
    fn leg_test() {
        let i = Leg::new();
        let j = Leg::new();
        let ii = i.prime();
        let iii1 = ii.prime();
        let iii2 = i.prime_by(2);
        println!("i :{:?}", i);
        println!("j :{:?}", j);
        println!("ii:{:?}", ii);
        assert_eq!(iii1, iii2);

        //let k = leg![i=>1,j=>2].unwrap();
        //println!("k: {:?}", k);
    }
}

// pub trait LegMap {
//     type Value;
// }

// pub trait MapLegAlloc<Map: LegMap>: LegAlloc {
//     fn translate_map(&self, indices: Map) -> Result<Vec<Map::Value>, LegError>;
// }

// pub trait LegAlloc: FromIterator<Self::Id> {
//     type Id;
//     fn len(&self) -> usize;
// }

// fn check_unique<Id: Eq>(legs: &[Id]) -> bool {
//     for i in 0..legs.len() {
//         for j in (i + 1)..legs.len() {
//             if legs[i] == legs[j] {
//                 return false;
//             }
//         }
//     }
//     true
// }

// impl<Id: Eq> LegAlloc<Id> {
//     pub fn from_raw(legs: Vec<Id>) -> Result<Self, Vec<Id>> {
//         if !check_unique(&legs) {
//             return Err(legs);
//         }
//         Ok(Self { legs })
//     }
//     pub fn into_raw(self) -> Vec<Id> {
//         self.legs
//     }

//     pub fn len(&self) -> usize {
//         self.legs.len()
//     }
//     pub fn contains(&self, leg: &Id) -> bool
//     where
//         Id: Eq,
//     {
//         self.legs.iter().find(|x| *x == leg).is_some()
//     }
//     pub fn replace(&mut self, old_leg: &Id, new_leg: Id) -> Result<Id, Id> {
//         if self.contains(&new_leg) {
//             return Err(new_leg);
//         }
//         match self.legs.iter().position(|x| x == old_leg) {
//             Some(idx) => {
//                 let old = core::mem::replace(&mut self.legs[idx], new_leg);
//                 Ok(old)
//             }
//             None => Err(new_leg),
//         }
//     }
// }

// /// leg set
// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct LegSet<Id: Eq> {
//     legs: Vec<Id>,
// }

// impl<Id: Eq> LegSet<Id> {
//     pub fn from_raw(legs: Vec<Id>) -> Result<Self, Vec<Id>> {
//         if !check_unique(&legs) {
//             return Err(legs);
//         }
//         Ok(Self { legs })
//     }
//     pub fn into_raw(self) -> Vec<Id> {
//         self.legs
//     }
// }

// /// leg reference set
// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct LegRefSet<'a, Id: Eq> {
//     leg_refs: Vec<&'a Id>,
// }

// impl<'a, Id: Eq> LegRefSet<'a, Id> {
//     pub fn from_raw(leg_refs: Vec<&'a Id>) -> Result<Self, Vec<&'a Id>> {
//         if !check_unique(&leg_refs) {
//             return Err(leg_refs);
//         }
//         Ok(Self { leg_refs })
//     }
//     pub fn into_raw(self) -> Vec<&'a Id> {
//         self.leg_refs
//     }
// }

// /// leg - value map
// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct LegMap<Id: Eq, V> {
//     legs: Vec<Id>,
//     values: Vec<V>,
// }

// /// leg reference - value map
// #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// pub struct LegRefMap<'a, Id: Eq, V> {
//     leg_refs: Vec<&'a Id>,
//     values: Vec<V>,
// }

// impl<Id: Eq, V> LegMap<Id, V> {
//     pub fn from_raw(legs: Vec<Id>, values: Vec<V>) -> Result<Self, (Vec<Id>, Vec<V>)> {
//         if !check_unique(&legs) || legs.len() != values.len() {
//             return Err((legs, values));
//         }
//         Ok(Self { legs, values })
//     }
//     pub fn into_raw(self) -> (Vec<Id>, Vec<V>) {
//         (self.legs, self.values)
//     }
// }

// impl<'a, Id: Eq, V> LegRefMap<'a, Id, V> {
//     pub fn from_raw(leg_refs: Vec<&'a Id>, values: Vec<V>) -> Result<Self, (Vec<&'a Id>, Vec<V>)> {
//         if !check_unique(&leg_refs) || leg_refs.len() != values.len() {
//             return Err((leg_refs, values));
//         }
//         Ok(Self { leg_refs, values })
//     }
//     pub fn into_raw(self) -> (Vec<&'a Id>, Vec<V>) {
//         (self.leg_refs, self.values)
//     }
// }

// // macro_rules! leg_map_elem {
// //     ($x)
// //     ($x:expr, $y:expr) => {
// //         LegVal($x, $y)
// //     };
// // }

// /// tuple of LegId and some value
// // #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// // pub struct LegVal<Id, I>(pub Id, pub I);

// // /// tuple of LegId reference and some value
// // #[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
// // pub struct LegRefVal<'a, Id: LegId, I>(pub &'a Id, pub I);

// // pub enum _LegWrapper<'a, T: LegId> {
// //     Owned(T),
// //     Ref(&'a T),
// // }
// // impl<'a, T: LegId> _LegWrapper<'a, T> {
// //     pub fn to_owned(self) -> T {
// //         match self {
// //             _LegWrapper::Owned(id) => id,
// //             _LegWrapper::Ref(id) => id.clone(),
// //         }
// //     }
// // }

// // impl<'a, T: LegId> From<T> for _LegWrapper<'a, T> {
// //     fn from(value: T) -> Self {
// //         _LegWrapper::Owned(value)
// //     }
// // }

// // impl<'a, T: LegId> From<&'a T> for _LegWrapper<'a, T> {
// //     fn from(value: &'a T) -> Self {
// //         _LegWrapper::Ref(value)
// //     }
// // }

// // #[macro_export]
// // /// LegVal list macro
// // // here we use converter wrapper to allow both Id and &Id
// // macro_rules! v {
// //     ( $( $x:expr => $y:expr),* ) => {
// //         [$($crate::core::tensor::LegVal($crate::core::tensor::_LegWrapper::from($x).to_owned(), $y)),*]
// //     };
// // }
// // #[macro_export]
// // /// LegRefVal list macro
// // // &($x) is available both for Id and &Id, thanks to auto deref
// // macro_rules! rv {
// //     ( $( $x:expr => $y:expr),* ) => {
// //         [$($crate::core::tensor::LegRefVal(&($x), $y)),*]
// //     };
// // }

// #[macro_export]
// macro_rules! leg {
//     ( $( $x:expr ),* ) => {
//         $crate::core::leg::LegSet::from_raw(::alloc::vec![$($x),*])
//     };
//     ( $( $x:expr => $y:expr ),* ) => {
//         $crate::core::leg::LegMap::from_raw(::alloc::vec![$($x),*],alloc::vec![$($y),*])
//     };
// }

// #[macro_export]
// macro_rules! leg_ref {
//     ( $( $x:expr ),* ) => {
//         $crate::core::leg::LegRefSet::from_raw(::alloc::vec![$($x),*])
//     };
//      ( $( $x:expr => $y:expr ),* ) => {
//         $crate::core::leg::LegRefMap::from_raw(::alloc::vec![$($x),*],alloc::vec![$($y),*])
//     };
// }

// impl<Id: Eq> LegAlloc<Id> {
//     pub fn pairs(&self, contract: &Self) -> Vec<(usize, usize)> {
//         let mut pairs = Vec::new();

//         for (i, leg) in self.legs.iter().enumerate() {
//             for (j, contract_leg) in contract.legs.iter().enumerate() {
//                 if leg == contract_leg {
//                     pairs.push((i, j));
//                     break;
//                 }
//             }
//         }
//         pairs
//     }

//     pub fn translate_set<'a>(&self, set: LegRefSet<'a, Id>) -> Result<Vec<usize>, LegError> {
//         set.leg_refs
//             .iter()
//             .map(|leg| self.legs.iter().position(|l| l == *leg))
//             .collect::<Option<Vec<_>>>()
//             .ok_or(LegError)
//     }

//     pub fn translate_map<'a, T>(&self, map: LegRefMap<'a, Id, T>) -> Result<Vec<T>, LegError> {
//         if let Some(idxs) = self
//             .legs
//             .iter()
//             .map(|leg| map.leg_refs.iter().position(|&l| l == leg))
//             .collect::<Option<Vec<_>>>()
//         {
//             let mut map = idxs
//                 .into_iter()
//                 .zip(map.values.into_iter())
//                 .collect::<BTreeMap<_, _>>();

//             Ok((0..self.legs.len())
//                 .map(|i| map.remove(&i).unwrap())
//                 .collect())
//         } else {
//             Err(LegError)
//         }
//     }
// }

// #[derive(Debug)]
// pub struct LegError;

// impl core::fmt::Display for LegError {
//     fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
//         write!(f, "leg error")
//     }
// }

// impl core::error::Error for LegError {
//     fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
//         None
//     }
// }

// #[cfg(test)]
// mod tests {
//     use std::vec;

//     use crate::core::leg::LegAlloc;

//     #[test]
//     fn leg_alloc_test() {
//         let legs = vec![1, 2, 3];
//         let mut leg_alloc = LegAlloc::from_raw(legs.clone()).unwrap();
//         assert_eq!(leg_alloc.len(), 3);
//         assert!(leg_alloc.contains(&1));
//         assert!(!leg_alloc.contains(&4));

//         let new_leg = 4;
//         let replaced_leg = leg_alloc.replace(&1, new_leg).unwrap();
//         assert_eq!(replaced_leg, 1);
//         assert!(leg_alloc.contains(&new_leg));
//         assert!(!leg_alloc.contains(&1));
//     }
// }
