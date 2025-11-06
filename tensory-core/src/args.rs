//! useful args when IDs involved

#[macro_export]
macro_rules! leg {
    ( $( $x:expr ),* ) => {
        $crate::args::LegSetArg::from_raw([$($x),*].into_iter())
    };
    ( $( $x:expr => $y:expr ),* ) => {
        $crate::args::_from_array_pair([$($x),*],[$($y),*])
    };
}

pub fn _from_array_pair<K, V, const N: usize>(
    keys: [K; N],
    values: [V; N],
) -> LegMapArg<<[K; N] as IntoIterator>::IntoIter, <[V; N] as IntoIterator>::IntoIter> {
    unsafe { LegMapArg::from_raw_unchecked(keys.into_iter(), values.into_iter()) }
}

#[macro_export]
macro_rules! ls {
    ( $( $x:expr ),* ) => {
        $crate::args::LegSetArg::from_raw([$($x),*].into_iter())
    };
}

#[macro_export]
macro_rules! lm {
    ( $( $x:expr => $y:expr ),* ) => {
        $crate::args::_from_array_pair([$($x),*],[$($y),*])
    };
}

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

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LegSetArg<T: ExactSizeIterator>(T);

impl<T: ExactSizeIterator> LegSetArg<T> {
    pub fn from_raw(legs: T) -> Self {
        Self(legs)
    }
    pub fn into_raw(self) -> T {
        self.0
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct LegMapArg<K: ExactSizeIterator, V: ExactSizeIterator>(K, V);

impl<K: ExactSizeIterator, V: ExactSizeIterator> LegMapArg<K, V> {
    pub fn from_raw(legs: K, values: V) -> Result<Self, (K, V)> {
        if legs.len() != values.len() {
            return Err((legs, values));
        }
        Ok(unsafe { Self::from_raw_unchecked(legs, values) })
    }
    pub unsafe fn from_raw_unchecked(legs: K, values: V) -> Self {
        Self(legs, values)
    }
    pub fn into_raw(self) -> (K, V) {
        (self.0, self.1)
    }
}

// #[macro_export]
// /// LegVal list macro
// // here we use converter wrapper to allow both Id and &Id
// macro_rules! v {
//     ( $( $x:expr => $y:expr),* ) => {
//         [$($crate::core::tensor::LegVal($crate::core::tensor::_LegWrapper::from($x).to_owned(), $y)),*]
//     };
// }
// #[macro_export]
// /// LegRefVal list macro
// // &($x) is available both for Id and &Id, thanks to auto deref
// macro_rules! rv {
//     ( $( $x:expr => $y:expr),* ) => {
//         [$($crate::core::tensor::LegRefVal(&($x), $y)),*]
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

#[derive(Debug)]
pub struct LegError;

impl core::fmt::Display for LegError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "leg error")
    }
}

impl core::error::Error for LegError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        None
    }
}

// #[cfg(test)]
// mod tests {
//     use std::vec;

//     use crate::leg::LegAlloc;

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
