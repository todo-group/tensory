use crate::{mapper::OverlayAxisMapping, port::PortError, repr::TensorTupleRepr};

/// Intermediate task struct for addition operation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AssignRepr<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> {
    src: S,
    dst: D,
    axis_mapping: [OverlayAxisMapping<2>; N],
}
impl<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> AssignRepr<N, S, D> {
    pub unsafe fn from_raw_unchecked(
        src: S,
        dst: D,
        axis_mapping: [OverlayAxisMapping<2>; N],
    ) -> Self {
        Self {
            src,
            dst,
            axis_mapping,
        }
    }
    pub fn from_raw(
        self,
        src: S,
        dst: D,
        axis_mapping: [OverlayAxisMapping<2>; N],
    ) -> Result<Self, PortError> {
        let n_s = src.naxeses();
        let n_d = dst.naxeses();
        let n = axis_mapping.each_ref().map(|m| m.naxes());
        if n_s != n || n_d != n {
            Err(PortError)
        } else {
            Ok(Self {
                src,
                dst,
                axis_mapping,
            })
        }
    }

    pub fn into_raw(self) -> (S, D, [OverlayAxisMapping<2>; N]) {
        (self.src, self.dst, self.axis_mapping)
    }
}

unsafe impl<const N: usize, S: TensorTupleRepr<N>, D: TensorTupleRepr<N>> TensorTupleRepr<N>
    for AssignRepr<N, S, D>
{
    fn naxeses(&self) -> [usize; N] {
        self.axis_mapping.each_ref().map(|m| m.naxes())
    }
}

// pub trait AssignTensorExt: ToTensorTuple {
//     /// Replace a ID of a leg of the tensor.
//     fn assign<T: ToTensorTuple<Mapper = Self::Mapper>>(
//         self,
//         dst: T,
//     ) -> Result<
//         Tensor<AssignRepr<Self::Repr, T::Repr>, Self::Mapper>,
//         <Self::Mapper as OverlayMapper<2>>::Err,
//     >
//     where
//         Self: Sized,
//         Self::Mapper: OverlayMapper<2>;
// }

// impl<S: ToTensorTuple> AssignTensorExt for S {
//     fn assign<D: ToTensorTuple<Mapper = Self::Mapper>>(
//         self,
//         dst: D,
//     ) -> Result<
//         Tensor<AssignRepr<Self::Repr, D::Repr>, Self::Mapper>,
//         <Self::Mapper as OverlayMapper<2>>::Err,
//     >
//     where
//         Self: Sized,
//         Self::Mapper: OverlayMapper<2>,
//     {
//         let (src, src_m) = self.to_tensor().into_raw();
//         let (dst, dst_m) = dst.to_tensor().into_raw();

//         let (mapper, axis_mapping) = OverlayMapper::<2>::overlay([src_m, dst_m])?;
//         Ok(unsafe {
//             Tensor::from_raw_unchecked(
//                 AssignRepr {
//                     src,
//                     dst,
//                     axis_mapping,
//                 },
//                 mapper,
//             )
//         })
//     }
// }
