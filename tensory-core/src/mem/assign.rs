//! Assignment of a lazy tensor into an existing tensor representation.

use crate::{
    mapper::{OverlayAxisMapping, OverlayMapper},
    repr::TensorTupleRepr,
    tensor::{TensorTuple, ToTensorTuple},
};
use alloc::vec::Vec;

/// Lazy assignment tensor representation.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct AssignRepr<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> {
    source: S,
    destination: D,
    axis_mappings: [OverlayAxisMapping<2>; N],
}

impl<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> AssignRepr<N, S, D> {
    /// Creates an assignment representation after validating its axis mappings.
    ///
    /// Each mapping is ordered as `[destination, source]`.
    pub fn from_raw(
        source: S,
        destination: D,
        axis_mappings: [OverlayAxisMapping<2>; N],
    ) -> Result<Self, (S, D, [OverlayAxisMapping<2>; N])> {
        let source_naxes = source.naxes_array();
        let destination_naxes = destination.naxes_array();
        let valid = axis_mappings.iter().enumerate().all(|(i, mapping)| {
            mapping.naxes() == source_naxes[i] && mapping.naxes() == destination_naxes[i]
        });

        if valid {
            Ok(unsafe { Self::from_raw_unchecked(source, destination, axis_mappings) })
        } else {
            Err((source, destination, axis_mappings))
        }
    }

    /// Creates an assignment representation without checking its axis mappings.
    ///
    /// # Safety
    ///
    /// For every index `i`, `axis_mappings[i]` must describe the axes of
    /// `source.naxes_array()[i]` and `destination.naxes_array()[i]`.
    pub unsafe fn from_raw_unchecked(
        source: S,
        destination: D,
        axis_mappings: [OverlayAxisMapping<2>; N],
    ) -> Self {
        Self {
            source,
            destination,
            axis_mappings,
        }
    }

    /// Decomposes the representation into its source, destination, and axis mappings.
    pub fn into_raw(self) -> (S, D, [OverlayAxisMapping<2>; N]) {
        (self.source, self.destination, self.axis_mappings)
    }
}

unsafe impl<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> TensorTupleRepr<N>
    for AssignRepr<N, S, D>
{
    fn naxes_array(&self) -> [usize; N] {
        self.axis_mappings.each_ref().map(|m| m.naxes())
    }
}

impl<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> crate::concept::task::IsTask
    for AssignRepr<N, S, D>
{
}

/// Provides lazy assignment for tensor tuples with matching leg sets.
pub trait AssignExt<const N: usize>: ToTensorTuple<N> {
    /// Creates an assignment task from `self` into `destination`.
    fn assign<D>(
        self,
        destination: D,
    ) -> Result<
        TensorTuple<N, AssignRepr<N, Self::Repr, D::Repr>, Self::Mapper>,
        <Self::Mapper as OverlayMapper<2>>::Err,
    >
    where
        D: ToTensorTuple<N, Mapper = Self::Mapper>,
        Self::Mapper: OverlayMapper<2>;
}

impl<const N: usize, T: ToTensorTuple<N>> AssignExt<N> for T {
    fn assign<D>(
        self,
        destination: D,
    ) -> Result<
        TensorTuple<N, AssignRepr<N, Self::Repr, D::Repr>, Self::Mapper>,
        <Self::Mapper as OverlayMapper<2>>::Err,
    >
    where
        D: ToTensorTuple<N, Mapper = Self::Mapper>,
        Self::Mapper: OverlayMapper<2>,
    {
        let (source, source_mappers) = self.to_tensor_tuple().into_raw();
        let (destination, destination_mappers) = destination.to_tensor_tuple().into_raw();

        let mut mappers = Vec::with_capacity(N);
        let mut axis_mappings = Vec::with_capacity(N);
        for (destination_mapper, source_mapper) in
            destination_mappers.into_iter().zip(source_mappers)
        {
            let (mapper, axis_mapping) =
                OverlayMapper::<2>::overlay([destination_mapper, source_mapper])?;
            mappers.push(mapper);
            axis_mappings.push(axis_mapping);
        }
        let mappers: [Self::Mapper; N] = match mappers.try_into() {
            Ok(mappers) => mappers,
            Err(_) => unreachable!(),
        };
        let axis_mappings: [OverlayAxisMapping<2>; N] = match axis_mappings.try_into() {
            Ok(axis_mappings) => axis_mappings,
            Err(_) => unreachable!(),
        };
        let task = unsafe { AssignRepr::from_raw_unchecked(source, destination, axis_mappings) };

        Ok(unsafe { TensorTuple::from_raw_unchecked(task, mappers) })
    }
}
