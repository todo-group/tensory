use alloc::vec::Vec;

// idea:
// use the only same functionalities Array and Vec have
//

pub trait Veccy {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<&Self::Item>;
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Item>;
    type Item;
}

pub struct Slicey<'a, V: Veccy> {
    len: usize,
    slice_provider: &'a V,
}
pub struct SliceyMut<'a, V: Veccy> {
    len: usize,
    slice_provider: &'a mut V,
}

pub struct ConstLen<const N: usize>;

pub struct DynLen(usize);

pub trait Arrayiy<T> {
    type Output;
}

impl<T, const N: usize> Arrayiy<T> for ConstLen<N> {
    type Output = [T; N];
}
impl<T> Arrayiy<T> for DynLen {
    type Output = Vec<T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_arrayiy() {}
}
