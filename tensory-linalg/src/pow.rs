use alloc::vec;

use tensory_core::{
    bound_tensor::{BoundTensor, Runtime, RuntimeError, ToBoundTensor},
    mapper::{
        AxisMapper, DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes,
        SplittyError,
    },
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of diagonalization operation: A= V*D*VC.
///
/// This trait is unsafe because the implementation must ensure that the list of `SvdAxisProvenance` is valid for the given tensor.
pub unsafe trait PowCtxImpl<A: TensorRepr, E> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs power operation on the tensors `a` with the given axes put into U and the rests into V, and returns the result tensors and the provenance of each axis.
    ///
    /// # Safety
    ///
    /// the user must ensure that the axes are valid for the given tensor.
    ///
    /// the implementor must ensure the list of `SvdAxisProvenance` is valid for the given tensor.
    unsafe fn pow_unchecked(
        self,
        a: A,
        power: E,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err>;
}

/// Safe version if `PowCtxImpl`.
/// The blanket implementation checks input and panic if the condition is not satisfied.
pub trait PowCtx<A: TensorRepr, E>: PowCtxImpl<A, E> {
    /// Safe version of `pow_unchecked`.
    fn pow(self, a: A, power: E, axes_split: EquivGroupedAxes<2>) -> Result<Self::Res, Self::Err>;
}

impl<C: PowCtxImpl<A, E>, A: TensorRepr, E> PowCtx<A, E> for C {
    fn pow(self, a: A, power: E, axes_split: EquivGroupedAxes<2>) -> Result<Self::Res, Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.pow_unchecked(a, power, axes_split) }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]

pub struct TensorPow<A: TensorRepr, E, M: AxisMapper> {
    a: A,
    power: E,
    legs: M,
    axes_split: EquivGroupedAxes<2>,
}

impl<A: TensorRepr, M: AxisMapper, C: PowCtx<A, E>, E> TensorTask<C> for TensorPow<A, E, M> {
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let res = unsafe { ctx.pow_unchecked(a, self.power, axes_split) }?;

        Ok(unsafe { Tensor::from_raw_unchecked(res, self.legs) })
    }
}
impl<A: TensorRepr, E, M: AxisMapper> TensorPow<A, E, M> {
    pub unsafe fn from_raw_unchecked(
        a: A,
        power: E,
        legs: M,
        axes_split: EquivGroupedAxes<2>,
    ) -> Self {
        TensorPow {
            a,
            power,
            legs,
            axes_split,
        }
    }
}

pub trait PowRuntime<A: TensorRepr, E>: Runtime {
    type Ctx: PowCtxImpl<A, E>;
    fn pow_ctx(&self) -> Self::Ctx;
}

pub trait TensorPowExt: ToTensor {
    fn pow<E, Q>(
        self,
        power: E,
        query: Q,
    ) -> Result<
        TensorPow<Self::Repr, E, Self::Mapper>,
        SplittyError<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>;
}

impl<T: ToTensor> TensorPowExt for T {
    fn pow<E, Q>(
        self,
        power: E,
        query: Q,
    ) -> Result<
        TensorPow<Self::Repr, E, Self::Mapper>,
        SplittyError<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
    {
        let (raw, legs) = self.to_tensor().into_raw();
        let (grouped, axes_split) = legs.equiv_split(query).map_err(SplittyError::Split)?;
        let [legs] = unsafe { grouped.decomp(DecompConf::from_raw_unchecked([0, 0], vec![])) }
            .map_err(SplittyError::Use)?;
        Ok(unsafe { TensorPow::from_raw_unchecked(raw, power, legs, axes_split) })
    }
}

pub trait BoundTensorPowExt: ToBoundTensor {
    fn pow<E, Q>(
        self,
        power: E,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as PowRuntime<Self::Repr, E>>::Ctx as PowCtxImpl<Self::Repr, E>>::Res,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeError<
            SplittyError<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            <<Self::Runtime as PowRuntime<Self::Repr,E>>::Ctx as PowCtxImpl<Self::Repr,E>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: PowRuntime<Self::Repr,E>;
}

impl<T: ToBoundTensor> BoundTensorPowExt for T {
    fn pow<E,Q>(
        self,
        power: E,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as PowRuntime<Self::Repr, E>>::Ctx as PowCtxImpl<Self::Repr, E>>::Res,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeError<
            SplittyError<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            <<Self::Runtime as PowRuntime<Self::Repr, E>>::Ctx as PowCtxImpl<Self::Repr, E>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: PowRuntime<Self::Repr, E>,
    {
        let (a, rt) = self.to_bound_tensor().into_raw();

        a.pow(power, query)
            .map_err(RuntimeError::Axis)?
            .with(rt.pow_ctx())
            .map_err(RuntimeError::Ctx)
            .map(|res| res.bind(rt))
    }
}
