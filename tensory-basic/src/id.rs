use core::{fmt, ops::Deref};

use alloc::string::String;
use uuid::Uuid;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
/// A globally unique 128-bit axis identifier.
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
    /// Creates a randomly generated identifier.
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
/// A string-backed axis tag.
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
    /// Creates a tag from its owned string value.
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
/// An axis identifier together with a prime level.
pub struct Prime<Id> {
    id: Id,
    plv: usize,
}
impl<Id> Default for Prime<Id>
where
    Id: Default,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<Id> Prime<Id> {
    /// Creates an unprimed identifier from `Id::default()`.
    pub fn new() -> Self
    where
        Id: Default,
    {
        Self {
            id: Id::default(),
            plv: 0,
        }
    }
    /// Creates an unprimed identifier from an existing ID.
    pub fn from(id: Id) -> Self {
        Self { id, plv: 0 }
    }
    /// Increases the prime level by one.
    pub fn prime(self) -> Self {
        self.prime_by(1)
    }
    /// Decreases the prime level by one, saturating at zero.
    pub fn deprime(self) -> Self {
        self.deprime_by(1)
    }
    /// Increases the prime level by `dplv`.
    pub fn prime_by(mut self, dplv: usize) -> Self {
        self.plv = self.plv.saturating_add(dplv);
        self
    }
    /// Decreases the prime level by `dplv`, saturating at zero.
    pub fn deprime_by(mut self, dplv: usize) -> Self {
        self.plv = self.plv.saturating_sub(dplv);
        self
    }
    /// Returns the current prime level.
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

        //let k = ls![i=>1,j=>2].unwrap();
        //println!("k: {:?}", k);
    }
}
