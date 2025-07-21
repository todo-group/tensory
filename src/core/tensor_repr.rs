use std::borrow::Borrow;

/// Minimal requirement to be a tensor repr
/// Raw tensors are treated as handles of resource allocation and management.
pub trait TensorRepr {
    fn len(&self) -> usize;
}

// pub trait Contract<Rhs: TensorRepr>: TensorRepr {
//     type Err;
//     type Res: TensorRepr;
//     fn contract_ref_ref(
//         &self,
//         rhs: &Rhs,
//         leg_pairs: &[(usize, usize)],
//     ) -> Result<(Self::Res, Vec<(bool, usize)>), Self::Err>;
//     fn contract_owned_ref(
//         self,
//         rhs: &Rhs,
//         leg_pairs: &[(usize, usize)],
//     ) -> Result<(Self::Res, Vec<(bool, usize)>), Self::Err>;
//     fn contract_ref_owned(
//         &self,
//         rhs: Rhs,
//         leg_pairs: &[(usize, usize)],
//     ) -> Result<(Self::Res, Vec<(bool, usize)>), Self::Err>;
//     fn contract_owned_owned(
//         self,
//         rhs: Rhs,
//         leg_pairs: &[(usize, usize)],
//     ) -> Result<(Self::Res, Vec<(bool, usize)>), Self::Err>;
// }

pub enum ContractionIndexProvenance {
    Lhs,
    Rhs,
}

pub trait ContractionContext<Lhs: TensorRepr, Rhs: TensorRepr> {
    type Res: TensorRepr;
    type Err;
    fn contract(
        self,
        lhs: Lhs,
        rhs: Rhs,
        idxs_contracted: &[(usize, usize)],
    ) -> Result<(Self::Res, Vec<(ContractionIndexProvenance, usize)>), Self::Err>;
}

pub trait SvdContext<A: TensorRepr, U: TensorRepr, S: TensorRepr, V: TensorRepr> {
    type Err;
    fn svd(
        self,
        a: A,
        u_legs: &[usize],
    ) -> Result<(U, S, V, Vec<(bool, usize)>, Vec<(bool, usize)>), Self::Err>;
}

// pub trait SvdAble: TensorRepr {}
// pub trait QrAble: TensorRepr {}
// pub trait RawLuAble: TensorRepr {}
// pub trait Raw: TensorRepr {}
// pub trait RawConjugatable: TensorRepr {}

pub trait ElementAccess: TensorRepr {
    type Index;
    type E;
    type Err: std::error::Error;
    fn get(
        &self,
        locs: impl IntoIterator<Item = impl Borrow<Self::Index>>,
    ) -> Result<&Self::E, Self::Err>;
    fn get_mut(
        &mut self,
        locs: impl IntoIterator<Item = impl Borrow<Self::Index>>,
    ) -> Result<&mut Self::E, Self::Err>;
}

// mat exponential
