use alloc::vec;

use tensory_core::{
    mapper::{
        AxisMapper, DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes,
        GroupMapper, GroupedAxes, SolveConf, SolveGroupedMapper, SplittyError,
    },
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of SVD operation.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait SolveEigCtxImpl<A: TensorRepr> {
    /// The type of the result tensor representation corresponding V.
    type V: TensorRepr; // axis order: a, <from A for V>
    /// The type of the result tensor representation corresponding D.
    type D: TensorRepr; // axis order: a, b

    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs SVD operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn solve_eig_unchecked(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::D, Self::V), Self::Err>;
}

/// Safe version if SvdContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait SolveEigCtx<A: TensorRepr>: SolveEigCtxImpl<A> {
    fn solve_eig(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::D, Self::V), Self::Err>;
}
impl<C: SolveEigCtxImpl<A>, A: TensorRepr> SolveEigCtx<A> for C {
    fn solve_eig(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<(Self::D, Self::V), Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.solve_eig_unchecked(a, axes_split) }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TensorSolveEig<A: TensorRepr, M: AxisMapper> {
    a: A,
    v_legs: M,
    d_legs: M,
    axes_split: EquivGroupedAxes<2>,
}

impl<A: TensorRepr, M: AxisMapper, C: SolveEigCtx<A>> TensorTask<C> for TensorSolveEig<A, M> {
    type Output = Result<(Tensor<C::V, M>, Tensor<C::D, M>), C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let (d, v) = unsafe { ctx.solve_eig_unchecked(a, axes_split) }?;

        Ok((
            unsafe { Tensor::from_raw_unchecked(v, self.v_legs) },
            unsafe { Tensor::from_raw_unchecked(d, self.d_legs) },
        ))
    }
}

pub trait TensorSolveEigExt<A: TensorRepr, M: AxisMapper>: Sized {
    fn solve_eig_with_more_ids<Q>(
        self,
        set: Q,
        v_vd_leg: M::Id,
        d_vd_leg: M::Id,
        d_dvinv_leg: M::Id,
    ) -> Result<
        TensorSolveEig<A, M>,
        SplittyError<M::Err, <M::Grouped as SolveGroupedMapper<2, 2>>::Err>,
    >
    where
        M: EquivGroupMapper<2, Q>,
        M::Grouped: SolveGroupedMapper<2, 2>;
    fn solve_eig<Q>(
        self,
        set: Q,
        vd_leg: M::Id,
        dvinv_leg: M::Id,
    ) -> Result<
        TensorSolveEig<A, M>,
        SplittyError<M::Err, <M::Grouped as SolveGroupedMapper<2, 2>>::Err>,
    >
    where
        M: EquivGroupMapper<2, Q>,
        M::Grouped: SolveGroupedMapper<2, 2>,
        M::Id: Clone;
}

impl<T: ToTensor> TensorSolveEigExt<T::Repr, T::Mapper> for T {
    fn solve_eig_with_more_ids<Q>(
        self,
        queue: Q,
        v_vd_leg: <T::Mapper as AxisMapper>::Id,
        d_vd_leg: <T::Mapper as AxisMapper>::Id,
        d_dvinv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorSolveEig<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs
            .equiv_split(queue)
            .map_err(|e| SplittyError::Split(e))?;
        let [v_legs, d_legs] = unsafe {
            grouped.solve(SolveConf::from_raw_unchecked(
                [[true, false], [false, false]],
                vec![(0, v_vd_leg), (1, d_vd_leg), (1, d_dvinv_leg)],
            ))
        }
        .map_err(|e| SplittyError::Use(e))?;
        Ok(TensorSolveEig {
            a: raw,
            d_legs,
            v_legs,
            axes_split,
        })
    }
    fn solve_eig<Q>(
        self,
        set: Q,
        vd_leg: <T::Mapper as AxisMapper>::Id,
        dvinv_leg: <T::Mapper as AxisMapper>::Id,
    ) -> Result<
        TensorSolveEig<T::Repr, T::Mapper>,
        SplittyError<
            <T::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<T::Mapper as EquivGroupMapper<2, Q>>::Grouped as SolveGroupedMapper<2, 2>>::Err,
        >,
    >
    where
        T::Mapper: EquivGroupMapper<2, Q>,
        <T::Mapper as EquivGroupMapper<2, Q>>::Grouped: SolveGroupedMapper<2, 2>,
        <T::Mapper as AxisMapper>::Id: Clone,
    {
        self.solve_eig_with_more_ids(set, vd_leg.clone(), vd_leg.clone(), dvinv_leg)
    }
}
