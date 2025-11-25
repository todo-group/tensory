use alloc::vec::Vec;

// idea:
// use the only same functionalities Array and Vec have
//

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
