#![no_std]
extern crate alloc;
#[cfg(test)]
extern crate std;

extern crate blas_src;

mod tenalg;

pub mod arith;

pub mod linalg;

pub mod cut_filter {
    pub use crate::tenalg::cut_filter::*;
}

use core::borrow::Borrow;

use rand::Rng;
use tensory_core::{
    args::{LegMapArg, LegSetArg},
    bound_tensor::Runtime,
    mapper::{AxisMapper, BuildableMapper, SynBuildableMapper},
    repr::{AsViewMutRepr, AsViewRepr},
    tensor::Tensor,
    utils::{
        axis_info::AxisInfoReprImpl,
        elem_get::{ElemGetMutReprImpl, ElemGetReprImpl},
    },
};

use alloc::vec::Vec;
use ndarray::{Array2, ArrayD, ArrayViewD, ArrayViewMutD};
use ndarray_linalg::{
    Lapack, Scalar, random, random_hermite, random_hermite_using, random_unitary,
    random_unitary_using, random_using,
};
use num_traits::{One, Zero};
use tensory_core::repr::TensorRepr;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseRepr<E> {
    data: ArrayD<E>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdDenseViewRepr<'a, E> {
    data: ArrayViewD<'a, E>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct NdDenseViewMutRepr<'a, E> {
    data: ArrayViewMutD<'a, E>,
}

impl<E> NdDenseRepr<E> {
    pub fn from_raw(data: ArrayD<E>) -> Self {
        Self { data }
    }
    pub fn random(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        Self {
            data: random(sizes),
        }
    }
    pub fn random_using(sizes: impl Iterator<Item = usize>, rng: &mut impl Rng) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        Self {
            data: random_using(sizes, rng),
        }
    }
    pub fn random_hermite(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        let permutor_pre = (0..sizes.len()).map(|x| 2 * x);
        let permutor: Vec<_> = permutor_pre
            .clone()
            .chain(permutor_pre.map(|x| x + 1))
            .collect();
        Self {
            data: random_hermite(full_size)
                .into_shape_with_order(sizes_dup)
                .unwrap()
                .permuted_axes(permutor),
        }
    }

    pub fn random_hermite_using(sizes: impl Iterator<Item = usize>, rng: &mut impl Rng) -> Self
    where
        E: Scalar,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        let permutor_pre = (0..sizes.len()).map(|x| 2 * x);
        let permutor: Vec<_> = permutor_pre
            .clone()
            .chain(permutor_pre.map(|x| x + 1))
            .collect();
        Self {
            data: random_hermite_using(full_size, rng)
                .into_shape_with_order(sizes_dup)
                .unwrap()
                .permuted_axes(permutor),
        }
    }

    pub fn random_unitary(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Scalar + Lapack,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        let permutor_pre = (0..sizes.len()).map(|x| 2 * x);
        let permutor: Vec<_> = permutor_pre
            .clone()
            .chain(permutor_pre.map(|x| x + 1))
            .collect();
        Self {
            data: random_unitary(full_size)
                .into_shape_with_order(sizes_dup)
                .unwrap()
                .permuted_axes(permutor),
        }
    }
    pub fn random_unitary_using(sizes: impl Iterator<Item = usize>, rng: &mut impl Rng) -> Self
    where
        E: Scalar + Lapack,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        let permutor_pre = (0..sizes.len()).map(|x| 2 * x);
        let permutor: Vec<_> = permutor_pre
            .clone()
            .chain(permutor_pre.map(|x| x + 1))
            .collect();
        Self {
            data: random_unitary_using(full_size, rng)
                .into_shape_with_order(sizes_dup)
                .unwrap()
                .permuted_axes(permutor),
        }
    }

    pub fn eye(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Clone + Zero + One,
    {
        let sizes: Vec<usize> = sizes.collect();
        let full_size = sizes.iter().product();
        let sizes_dup = sizes
            .iter()
            .cloned()
            .chain(sizes.iter().cloned())
            .collect::<Vec<_>>();
        let permutor_pre = (0..sizes.len()).map(|x| 2 * x);
        let permutor: Vec<_> = permutor_pre
            .clone()
            .chain(permutor_pre.map(|x| x + 1))
            .collect();
        Self {
            data: Array2::<E>::eye(full_size)
                .into_shape_with_order(sizes_dup)
                .unwrap()
                .permuted_axes(permutor),
        }
    }

    pub fn zero(sizes: impl Iterator<Item = usize>) -> Self
    where
        E: Clone + Zero,
    {
        let sizes: Vec<usize> = sizes.collect();
        Self {
            data: ArrayD::zeros(sizes),
        }
    }
    pub fn map<E2, F: FnMut(&E) -> E2>(&self, f: F) -> NdDenseRepr<E2> {
        NdDenseRepr {
            data: self.data.map(f),
        }
    }
}

unsafe impl<'a, E: 'a> AsViewRepr<'a> for NdDenseRepr<E> {
    type View = NdDenseViewRepr<'a, E>;
    fn view(&'a self) -> Self::View {
        NdDenseViewRepr {
            data: self.data.view(),
        }
    }
}
unsafe impl<'a, E: 'a> AsViewMutRepr<'a> for NdDenseRepr<E> {
    type ViewMut = NdDenseViewMutRepr<'a, E>;
    fn view_mut(&'a mut self) -> Self::ViewMut {
        NdDenseViewMutRepr {
            data: self.data.view_mut(),
        }
    }
}

unsafe impl<E> TensorRepr for NdDenseRepr<E> {
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
unsafe impl<E> TensorRepr for NdDenseViewRepr<'_, E> {
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
unsafe impl<E> TensorRepr for NdDenseViewMutRepr<'_, E> {
    fn naxes(&self) -> usize {
        self.data.shape().len()
    }
}
impl<'a, E> AxisInfoReprImpl<'a> for NdDenseRepr<E> {
    type AxisInfo = usize;

    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
        self.data.shape()[i]
    }
}

impl<'a, E> AxisInfoReprImpl<'a> for NdDenseViewRepr<'a, E> {
    type AxisInfo = usize;

    unsafe fn axis_info_unchecked(&self, i: usize) -> Self::AxisInfo {
        self.data.shape()[i]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdDenseReprError;
impl core::error::Error for NdDenseReprError {}
impl core::fmt::Display for NdDenseReprError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DenseRepr operation failed")
    }
}

impl<E> ElemGetReprImpl for NdDenseRepr<E> {
    type Index = usize;
    type E = E;
    type Err = NdDenseReprError;

    unsafe fn get_unchecked(&self, indices: Vec<Self::Index>) -> Result<&Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

impl<E> ElemGetMutReprImpl for NdDenseRepr<E> {
    unsafe fn get_mut_unchecked(
        &mut self,
        indices: Vec<Self::Index>,
    ) -> Result<&mut Self::E, Self::Err> {
        let locv: Vec<usize> = indices.into_iter().map(|i| *i.borrow()).collect();
        self.data.get_mut(locv.as_slice()).ok_or(NdDenseReprError)
    }
}

// unsafe impl<E: Scalar> ConjCtx<NdDenseViewRepr<'_, E>> for () {
//     type Res = NdDenseRepr<E>;

//     type Err = TenalgError;

//     fn conjugate(
//         self,
//         a: NdDenseViewRepr<'_, E>,
//     ) -> core::result::Result<(NdDenseRepr<E>, Vec<ConjugationAxisProvenance>), TenalgError> {
//         Ok((
//             NdDenseRepr {
//                 data: conj(&a.data)?,
//             },
//             (0..a.data.shape().len())
//                 .map(|i| ConjugationAxisProvenance(i))
//                 .collect(),
//         ))
//     }
// }

pub type NdDenseTensor<E, B> = Tensor<NdDenseRepr<E>, B>;
pub type NdDenseViewTensor<'a, E, B> = Tensor<NdDenseViewRepr<'a, E>, B>;
pub type NdDenseViewMutTensor<'a, E, B> = Tensor<NdDenseViewMutRepr<'a, E>, B>;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Error)]
pub enum NdDenseFromArrayError<E> {
    #[error("The number of axes of array and leg set do not match")]
    AxisNum,
    #[error("Mapper build failed:{0}")]
    MapperBuild(#[from] E),
}
// impl core::error::Error for NdDenseFromArrayError {}
// impl core::fmt::Display for NdDenseFromArrayError {
//     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//         write!(f, "DenseTensor operation failed")
//     }
// }

pub trait NdDenseTensorExt<E, M: AxisMapper>: Sized {
    fn from_array<K: ExactSizeIterator + Iterator<Item = M::Id>>(
        array: ArrayD<E>,
        set: LegSetArg<K>,
    ) -> Result<Self, NdDenseFromArrayError<M::Err>>
    where
        M: BuildableMapper<K>;
    fn zero<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        M: BuildableMapper<K>;
    fn eye<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Clone + One + Zero,
        M: SynBuildableMapper<K>;
    fn random<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;
    fn random_using<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>;

    fn random_hermite<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: SynBuildableMapper<K>;
    fn random_hermite_using<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: SynBuildableMapper<K>;

    fn random_unitary<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar + Lapack,
        M: SynBuildableMapper<K>;
    fn random_unitary_using<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar + Lapack,
        M: SynBuildableMapper<K>;

    fn map<E2, F: FnMut(&E) -> E2>(&self, f: F) -> Tensor<NdDenseRepr<E2>, M>
    where
        M: Clone;
}
impl<E, M: AxisMapper> NdDenseTensorExt<E, M> for NdDenseTensor<E, M> {
    fn from_array<K: ExactSizeIterator + Iterator<Item = M::Id>>(
        array: ArrayD<E>,
        set: LegSetArg<K>,
    ) -> Result<Self, NdDenseFromArrayError<M::Err>>
    where
        M: BuildableMapper<K>,
    {
        let set = set.into_raw();
        if array.ndim() == set.len() {
            let mapper = M::build(set)?;
            Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::from_raw(array), mapper) })
        } else {
            Err(NdDenseFromArrayError::AxisNum)
        }
    }
    fn zero<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Clone + Zero,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();

        let mapper = M::build(k)?;

        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::zero(v), mapper) })
    }
    fn eye<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Clone + One + Zero,
        M: SynBuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::syn_build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::eye(v), mapper) })
    }

    fn random<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random(v), mapper) })
    }
    fn random_using<
        K: ExactSizeIterator + Iterator<Item = M::Id>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as BuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: BuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random_using(v, rng), mapper) })
    }

    fn random_hermite<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: SynBuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::syn_build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random_hermite(v), mapper) })
    }
    fn random_hermite_using<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar,
        M: SynBuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::syn_build(k)?;
        Ok(
            unsafe {
                Tensor::from_raw_unchecked(NdDenseRepr::random_hermite_using(v, rng), mapper)
            },
        )
    }

    fn random_unitary<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar + Lapack,
        M: SynBuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::syn_build(k)?;
        Ok(unsafe { Tensor::from_raw_unchecked(NdDenseRepr::random_unitary(v), mapper) })
    }
    fn random_unitary_using<
        K: ExactSizeIterator + Iterator<Item = [M::Id; 2]>,
        V: ExactSizeIterator + Iterator<Item = usize>,
    >(
        map: LegMapArg<K, V>,
        rng: &mut impl Rng,
    ) -> Result<Self, <M as SynBuildableMapper<K>>::Err>
    where
        E: Scalar + Lapack,
        M: SynBuildableMapper<K>,
    {
        let (k, v) = map.into_raw();
        let mapper = M::syn_build(k)?;
        Ok(
            unsafe {
                Tensor::from_raw_unchecked(NdDenseRepr::random_unitary_using(v, rng), mapper)
            },
        )
    }

    fn map<E2, F: FnMut(&E) -> E2>(&self, mut f: F) -> Tensor<NdDenseRepr<E2>, M>
    where
        M: Clone,
    {
        unsafe { Tensor::from_raw_unchecked(self.repr().map(&mut f), self.mapper().clone()) }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct NdRuntime;
unsafe impl Runtime for NdRuntime {}

#[cfg(all(test, not(feature = "__test-build")))]
compile_error!("test build requires __test-build feature to be enabled for lin-dev-deps");
