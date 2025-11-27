use alloc::vec;

use tensory_core::{
    bound_tensor::{BoundTensor, Runtime, ToBoundTensor},
    mapper::{
        AxisMapper, DecompConf, DecompGroupedMapper, EquivGroupMapper, EquivGroupedAxes,
        GroupMapper, SplittyError,
    },
    prelude::RuntimeError,
    repr::TensorRepr,
    tensor::{Tensor, TensorTask, ToTensor},
};

/// Raw context of exponentiation operation.
///
/// # Safety
///
/// The implementor MUST ensure that the result tensor has the specified "axis structure" by the axes_split.
pub unsafe trait ExpCtxImpl<A: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs exp operation on the tensors `a` with the given axes split.
    ///
    /// # Safety
    ///
    /// The user MUST ensure that the `axes_split` are valid for the given tensor.
    unsafe fn exp_unchecked(
        self,
        a: A,
        axes_split: EquivGroupedAxes<2>,
    ) -> Result<Self::Res, Self::Err>;
}

/// Safe version if ExpContextImpl.
///
/// The blanket implementation checks both input and output.
pub trait ExpCtx<A: TensorRepr>: ExpCtxImpl<A> {
    fn exp(self, a: A, axes_split: EquivGroupedAxes<2>) -> Result<Self::Res, Self::Err>;
}
impl<C: ExpCtxImpl<A>, A: TensorRepr> ExpCtx<A> for C {
    fn exp(self, a: A, axes_split: EquivGroupedAxes<2>) -> Result<Self::Res, Self::Err> {
        if a.naxes() != axes_split.len() {
            panic!("Incompatible tensor dimensions");
        }
        unsafe { self.exp_unchecked(a, axes_split) }
    }
}

pub struct TensorExp<A: TensorRepr, M: AxisMapper> {
    a: A,
    legs: M,
    axes_split: EquivGroupedAxes<2>,
}

impl<A: TensorRepr, M: AxisMapper, C: ExpCtx<A>> TensorTask<C> for TensorExp<A, M> {
    type Output = Result<Tensor<C::Res, M>, C::Err>;

    fn with(self, ctx: C) -> Self::Output {
        let a = self.a;
        let axes_split = self.axes_split;

        let res = unsafe { ctx.exp_unchecked(a, axes_split) }?;

        Ok(unsafe { Tensor::from_raw_unchecked(res, self.legs) })
    }
}
impl<A: TensorRepr, M: AxisMapper> TensorExp<A, M> {
    pub unsafe fn from_raw_unchecked(a: A, legs: M, axes_split: EquivGroupedAxes<2>) -> Self {
        TensorExp {
            a,
            legs,
            axes_split,
        }
    }
}

pub trait ExpRuntime<A: TensorRepr>: Runtime {
    type Ctx: ExpCtxImpl<A>;
    fn exp_ctx(&self) -> Self::Ctx;
}

pub trait TensorExpExt: ToTensor {
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        TensorExp<Self::Repr, Self::Mapper>,
        SplittyError<
            <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
            <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>;
}

impl<T: ToTensor> TensorExpExt for T {
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        TensorExp<Self::Repr, Self::Mapper>,
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
        Ok(unsafe { TensorExp::from_raw_unchecked(raw, legs, axes_split) })
    }
}

pub trait BoundTensorExpExt: ToBoundTensor {
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as ExpRuntime<Self::Repr>>::Ctx as ExpCtxImpl<Self::Repr>>::Res,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeError<
            SplittyError<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            <<Self::Runtime as ExpRuntime<Self::Repr>>::Ctx as ExpCtxImpl<Self::Repr>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: ExpRuntime<Self::Repr>;
}

impl<T: ToBoundTensor> BoundTensorExpExt for T {
    fn exp<Q>(
        self,
        query: Q,
    ) -> Result<
        BoundTensor<
            <<Self::Runtime as ExpRuntime<Self::Repr>>::Ctx as ExpCtxImpl<Self::Repr>>::Res,
            Self::Mapper,
            Self::Runtime,
        >,
        RuntimeError<
            SplittyError<
                <Self::Mapper as EquivGroupMapper<2, Q>>::Err,
                <<Self::Mapper as EquivGroupMapper<2, Q>>::Grouped as DecompGroupedMapper<2, 1>>::Err,
            >,
            <<Self::Runtime as ExpRuntime<Self::Repr>>::Ctx as ExpCtxImpl<Self::Repr>>::Err,
        >,
    >
    where
        Self::Mapper: EquivGroupMapper<2, Q>,
        <Self::Mapper as EquivGroupMapper<2, Q>>::Grouped: DecompGroupedMapper<2, 1>,
        Self::Runtime: ExpRuntime<Self::Repr>,
    {
        let (a, rt) = self.to_bound_tensor().into_raw();

        a.exp(query)
            .map_err(RuntimeError::Axis)?
            .with(rt.exp_ctx())
            .map_err(RuntimeError::Ctx)
            .map(|res| res.bind(rt))
    }
}
