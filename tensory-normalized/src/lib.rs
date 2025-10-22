use tensory_core::{
    arith::{MulCtxImpl, TensorMul},
    mapper::{AxisMapper, ConnectAxisOrigin},
    repr::TensorRepr,
    tensor::Tensor,
};

struct NormalizedRepr<R: TensorRepr, N: Normalizer<R>> {
    repr: R,
    normalizer: N,
}

trait Normalizer<R: TensorRepr> {}

unsafe impl<R: TensorRepr, N: Normalizer<R>> TensorRepr for NormalizedRepr<R, N> {
    fn naxes(&self) -> usize {
        self.repr.naxes()
    }
}

impl<R: TensorRepr, N: Normalizer<R>> NormalizedRepr<R, N> {
    pub fn new(repr: R, normalizer: N) -> Self {
        Self { repr, normalizer }
    }
}

struct NormalizedCtx<C>(C);

pub trait NormalizedMul: Sized {
    type Lhs: TensorRepr;
    type Rhs: TensorRepr;
    type Mapper: AxisMapper;
    type Normalizer: Normalizer<Self::Lhs> + Normalizer<Self::Rhs>;
    fn with_normalized<C: MulCtxImpl<Self::Lhs, Self::Rhs>>(
        self,
        context: C,
    ) -> Result<Tensor<NormalizedRepr<C::Res, Self::Normalizer>, Self::Mapper>, C::Err>
    where
        Self::Normalizer: Normalizer<Self::Lhs> + Normalizer<Self::Rhs> + Normalizer<C::Res>;
}
impl<L: TensorRepr, R: TensorRepr, N: Normalizer<L> + Normalizer<R>, M: AxisMapper> NormalizedMul
    for TensorMul<NormalizedRepr<L, N>, NormalizedRepr<R, N>, M>
{
    type Lhs = L;
    type Rhs = R;
    type Mapper = M;
    type Normalizer = N;

    fn with_normalized<C: MulCtxImpl<L, R>>(
        self,
        context: C,
    ) -> Result<Tensor<NormalizedRepr<C::Res, Self::Normalizer>, Self::Mapper>, C::Err>
    where
        Self::Normalizer: Normalizer<L> + Normalizer<R> + Normalizer<C::Res>,
    {
        todo!()
        //self.with(NormalizedCtx(context))
    }
}

unsafe impl<
    C: MulCtxImpl<Lhs, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Normalizer<Lhs> + Normalizer<Rhs> + Normalizer<C::Res>,
> MulCtxImpl<NormalizedRepr<Lhs, N>, NormalizedRepr<Rhs, N>> for NormalizedCtx<C>
{
    type Res = NormalizedRepr<C::Res, N>;
    type Err = C::Err;

    unsafe fn mul_unchecked(
        self,
        lhs: NormalizedRepr<Lhs, N>,
        rhs: NormalizedRepr<Rhs, N>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        // let res = unsafe { self.0.mul_unchecked(lhs.repr, rhs.repr, axis_origin) }?;
        // Ok(NormalizedRepr {
        //     repr: res,
        //     normalizer: lhs.normalizer * rhs.normalizer,
        // })
        todo!()
    }
}
