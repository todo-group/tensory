use alloc::vec;

use tensory_core::{
    mapper::{
        AxisMapper, DecompConf, DecompError, DecompGroupedMapper, EquivGroupMapper,
        EquivGroupedAxes, GroupMapper, GroupedAxes,
    },
    repr::TensorRepr,
    tensor::{Tensor, ToTensor},
};

/// Raw context of SVD operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait EighContextImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding VC.
    type VC: TensorRepr; // axis order: a, <from A for VC>
    /// The type of the result tensor representation corresponding D.
    type D: TensorRepr; // axis order: a, b
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr; // axis order: b, <from A for V>
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs SVD operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn eigh_unchecked(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::VC, Self::D, Self::V), Self::Err>;
}

/// Safe version if SvdContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait EighContext<A: TensorRepr>: EighContextImpl<A> {
    fn eigh(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::VC, Self::D, Self::V), Self::Err>;
}
impl<C: EighContextImpl<A>, A: TensorRepr> EighContext<A> for C {
    fn eigh(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::VC, Self::D, Self::V), Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.eigh_unchecked(a, axes_split) }
    }
}

pub struct TensorEigh<A: TensorRepr, B: AxisMapper> {
    a: A,
    vc_legs: B,
    d_legs: B,
    v_legs: B,
    axes_split: EquivGroupedAxes<2>,
}

// pub unsafe trait SvdLegAlloc<Set>: LegAlloc {
//     type Intermediate;
//     fn extract(
//         a: Self,
//         u_legs: Set,
//         u_us_leg: Self::Id,
//         s_us_leg: Self::Id,
//         s_sv_leg: Self::Id,
//         v_sv_leg: Self::Id,
//     ) -> (Self::Intermediate, Vec<usize>);
//     unsafe fn merge(
//         intermediate: Self::Intermediate,
//         u_provenance: Vec<SvdIsometryAxisProvenance>,
//         s_provenance: SvdSingularAxisOrder,
//         v_provenance: Vec<SvdIsometryAxisProvenance>,
//     ) -> (Self, Self, Self)
//     where
//         Self: Sized;
// }

impl<A: TensorRepr, M: AxisMapper> TensorEigh<A, M> {
    // pub fn new<Q>(
    //     a: Tensor<A, B>,
    //     queue: Q,
    //     s_us_leg: B::Id,
    //     s_sv_leg: B::Id,
    //     v_sv_leg: B::Id,
    // ) -> Self {
    //     let (raw, legs) = a.into_raw();

    //     let (intermediate, u_axes) =
    //         LA::extract(legs, u_legs, u_us_leg, s_us_leg, s_sv_leg, v_sv_leg);

    //     Self {
    //         a: raw,
    //         intermediate,
    //         u_axes: u_axes,
    //     }
    // }
    pub fn with<C: EighContext<A>>(
        self,
        context: C,
    ) -> Result<(Tensor<C::VC, M>, Tensor<C::D, M>, Tensor<C::V, M>), C::Err> {
        let a = self.a;
        let axes_split = self.axes_split;

        let (u, s, v) = unsafe { context.eigh_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(u, self.vc_legs) },
            unsafe { Tensor::from_raw_unchecked(s, self.d_legs) },
            unsafe { Tensor::from_raw_unchecked(v, self.v_legs) },
        ))
    }
}

pub trait TensorEighExt<A: TensorRepr, B: AxisMapper>: Sized {
    fn eigh_with_more_ids<Q>(
        self,
        set: Q,
        vc_vcd_leg: B::Id,
        d_vcd_leg: B::Id,
        d_dv_leg: B::Id,
        v_dv_leg: B::Id,
    ) -> Result<TensorEigh<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        B: EquivGroupMapper<2, Q>,
        B::Grouped: DecompGroupedMapper<2, 3>;
    fn eigh<Q>(
        self,
        set: Q,
        vcd_leg: B::Id,
        dv_leg: B::Id,
    ) -> Result<TensorEigh<A, B>, DecompError<B::Err, <B::Grouped as DecompGroupedMapper<2, 3>>::Err>>
    where
        B: EquivGroupMapper<2, Q>,
        B::Grouped: DecompGroupedMapper<2, 3>,
        B::Id: Clone;
}

impl<T: ToTensor> TensorEighExt<T::Repr, T::Mapper> for T {
    fn eigh_with_more_ids<Q>(
        self,
        queue: Q,
        vc_vcd_leg: <T::Mapper as AxisMapper>::Id,
        d_vcd_leg: <T::Mapper as AxisMapper>::Id,
        d_dv_leg: <T::Mapper as AxisMapper>::Id,
        v_dv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorEigh<T::Repr, T::Mapper>,
        DecompError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.equiv_split(queue).map_err(|e| DecompError::Split(e))?;
        let [vc_legs, d_legs, v_legs] = unsafe {
            grouped.decomp(DecompConf::from_raw_unchecked(
                [0, 2],
                vec![
                    ((0, vc_vcd_leg), (1, d_vcd_leg)),
                    ((1, d_dv_leg), (2, v_dv_leg)),
                ],
            ))
        }
        .map_err(|e| DecompError::Decomp(e))?;
        Ok(TensorEigh {
            a: raw,
            vc_legs,
            d_legs,
            v_legs,
            axes_split,
        })
    }
    fn eigh<Q>(
        self,
        set: Q,
        vcd_leg: <T::Mapper as AxisMapper>::Id,
        dv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorEigh<T::Repr, T::Mapper>,
        DecompError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 3>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 3>,
        <T::Mapper as AxisMapper>::Id: Clone,
    {
        self.eigh_with_more_ids(set, vcd_leg.clone(), vcd_leg, dv_leg.clone(), dv_leg)
    }
}
