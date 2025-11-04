use tensory_core::{
    arith::{MulCtxImpl, TensorMul},
    mapper::{AxisMapper, ConnectAxisOrigin},
    repr::TensorRepr,
    tensor::{Tensor, TensorTask},
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

pub trait TensorNormalizedTask<C> {
    type Output;
    fn with_normalized(self, inner_ctx: C) -> Self::Output;
}

impl<T: TensorTask<NormalizedCtx<C>>, C> TensorNormalizedTask<C> for T {
    type Output = T::Output;
    fn with_normalized(self, inner_ctx: C) -> Self::Output {
        self.with(NormalizedCtx(inner_ctx))
    }
}

// pub trait NormalizedMul<C:MulCtxImpl<Self::Lhs, Self::Rhs>>: Sized {
//     type Lhs: TensorRepr;
//     type Rhs: TensorRepr;
//     type Mapper: AxisMapper;
//     type Normalizer: Normalizer<Self::Lhs> + Normalizer<Self::Rhs> + Normalizer<C::Res>;
//     fn with_normalized(
//         self,
//         context: C,
//     ) -> Result<Tensor<NormalizedRepr<C::Res, Self::Normalizer>, Self::Mapper>, C::Err>;
// }
// impl<L: TensorRepr, R: TensorRepr, N: Normalizer<L> + Normalizer<R> + Normalizer<C::Res>, M: AxisMapper,C:MulCtxImpl<L,R>> NormalizedMul<C>
//     for TensorMul<NormalizedRepr<L, N>, NormalizedRepr<R, N>, M>
// {
//     type Lhs = L;
//     type Rhs = R;
//     type Mapper = M;
//     type Normalizer = N;

//     fn with_normalized(
//         self,
//         context: C,
//     ) -> Result<Tensor<NormalizedRepr<C::Res, N>, M>, C::Err>
//     {
//         self.with(NormalizedCtx(context))
//     }
// }

//a*b .with_normalized(ctx);

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
        let res = unsafe { self.0.mul_unchecked(lhs.repr, rhs.repr, axis_origin) }?;
        Ok(NormalizedRepr {
            repr: res,
            normalizer: lhs.normalizer * rhs.normalizer,
        })
        //todo!()
    }
}
