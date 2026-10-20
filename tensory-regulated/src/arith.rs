use core::marker::PhantomData;

use tensory_core::{
    arith::{
        AddOp, AddRepr, LeftScalarDivOp, LeftScalarDivRepr, LeftScalarMulOp,
        LeftScalarMulRepr, MulRepr, RightScalarDivOp, RightScalarDivRepr,
        RightScalarMulOp, RightScalarMulRepr, SubOp, SubRepr,
    },
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::Context,
    },
    op::AxisAllocation,
    repr::{ReprContext, TensorRepr, TensorTupleRepr},
};

use crate::{CoefficientRepr, RegulatedCtx, RegulatedRepr, Regulator, ScalarRegulator};

pub unsafe trait WeightedAddCtxImpl<
    LhsCo,
    Lhs: TensorRepr,
    RhsCo,
    Rhs: TensorRepr,
> {
    type Res: TensorRepr;
    type Err;

    unsafe fn weighted_add_unchecked(
        self,
        lhs_coeff: LhsCo,
        lhs: Lhs,
        rhs_coeff: RhsCo,
        rhs: Rhs,
        axis_mapping: AxisAllocation<2>,
    ) -> Result<Self::Res, Self::Err>;
}

pub unsafe trait WeightedSubCtxImpl<
    LhsCo,
    Lhs: TensorRepr,
    RhsCo,
    Rhs: TensorRepr,
> {
    type Res: TensorRepr;
    type Err;

    unsafe fn weighted_sub_unchecked(
        self,
        lhs_coeff: LhsCo,
        lhs: Lhs,
        rhs_coeff: RhsCo,
        rhs: Rhs,
        axis_mapping: AxisAllocation<2>,
    ) -> Result<Self::Res, Self::Err>;
}

impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    Context<Mk, AddRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    C: WeightedAddCtxImpl<N::Scalar, Lhs, N::Scalar, Rhs, Res = Inner, Err = Err>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Output = Result<RegulatedRepr<N::Res, Co, N>, Err>;

    fn execute(
        self,
        task: AddRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_mapping], AddOp) = task.into_raw();
        let (lhs, lhs_coeff) = lhs.into_raw();
        let (rhs, rhs_coeff) = rhs.into_raw();
        let (coeff, [lhs_coeff, rhs_coeff]) = Co::factorize([lhs_coeff, rhs_coeff]);
        let repr = unsafe {
            self.0.weighted_add_unchecked(lhs_coeff, lhs, rhs_coeff, rhs, axis_mapping)
        }?;
        let (repr, coeff_more) = N::regulate(repr);
        Ok(RegulatedRepr {
            repr,
            coeff: coeff.mul([coeff_more]),
            _reg: PhantomData,
        })
    }
}

unsafe impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    ReprContext<Mk, 1, AddRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    C: WeightedAddCtxImpl<N::Scalar, Lhs, N::Scalar, Rhs, Res = Inner, Err = Err>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Repr = RegulatedRepr<N::Res, Co, N>;
    type CType = Resulting<Raw, Err>;
}

impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    Context<Mk, SubRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    C: WeightedSubCtxImpl<N::Scalar, Lhs, N::Scalar, Rhs, Res = Inner, Err = Err>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Output = Result<RegulatedRepr<N::Res, Co, N>, Err>;

    fn execute(
        self,
        task: SubRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_mapping], SubOp) = task.into_raw();
        let (lhs, lhs_coeff) = lhs.into_raw();
        let (rhs, rhs_coeff) = rhs.into_raw();
        let (coeff, [lhs_coeff, rhs_coeff]) = Co::factorize([lhs_coeff, rhs_coeff]);
        let repr = unsafe {
            self.0.weighted_sub_unchecked(lhs_coeff, lhs, rhs_coeff, rhs, axis_mapping)
        }?;
        let (repr, coeff_more) = N::regulate(repr);
        Ok(RegulatedRepr {
            repr,
            coeff: coeff.mul([coeff_more]),
            _reg: PhantomData,
        })
    }
}

unsafe impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    ReprContext<Mk, 1, SubRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    C: WeightedSubCtxImpl<N::Scalar, Lhs, N::Scalar, Rhs, Res = Inner, Err = Err>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Repr = RegulatedRepr<N::Res, Co, N>;
    type CType = Resulting<Raw, Err>;
}

impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    Context<Mk, MulRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorTupleRepr<1>,
    Rhs: TensorTupleRepr<1>,
    C: ReprContext<Mk, 1, MulRepr<1, Lhs, Rhs>, Repr = Inner, CType = Resulting<Raw, Err>>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Output = Result<RegulatedRepr<N::Res, Co, N>, Err>;

    fn execute(
        self,
        task: MulRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>,
    ) -> Self::Output {
        let (lhs, rhs, [axis_mapping], op) = task.into_raw();
        let (lhs, lhs_coeff) = lhs.into_raw();
        let (rhs, rhs_coeff) = rhs.into_raw();
        let task = unsafe { MulRepr::from_raw_unchecked((lhs, rhs, [axis_mapping], op)) };
        let repr = self.0.execute(task)?;
        let (repr, coeff_more) = N::regulate(repr);
        let coeff = Co::merge([lhs_coeff, rhs_coeff]).mul([coeff_more]);
        Ok(RegulatedRepr {
            repr,
            coeff,
            _reg: PhantomData,
        })
    }
}

unsafe impl<Mk, C, Lhs, Rhs, Co, N, Inner, Err>
    ReprContext<Mk, 1, MulRepr<1, RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>>>
    for RegulatedCtx<C>
where
    Lhs: TensorTupleRepr<1>,
    Rhs: TensorTupleRepr<1>,
    C: ReprContext<Mk, 1, MulRepr<1, Lhs, Rhs>, Repr = Inner, CType = Resulting<Raw, Err>>,
    Inner: TensorTupleRepr<1>,
    N: Regulator<Inner>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
{
    type Repr = RegulatedRepr<N::Res, Co, N>;
    type CType = Resulting<Raw, Err>;
}

macro_rules! impl_regulated_scalar {
    ($repr:ident, $op:ident, $combine:ident) => {
        impl<Mk, C, A, E, Co, N, Inner, Err>
            Context<Mk, $repr<1, RegulatedRepr<A, Co, N>, E>> for RegulatedCtx<C>
        where
            A: TensorTupleRepr<1>,
            C: ReprContext<Mk, 1, $repr<1, A, E>, Repr = Inner, CType = Resulting<Raw, Err>>,
            Inner: TensorTupleRepr<1>,
            N: ScalarRegulator<E>,
            Co: CoefficientRepr<Scalar = N::Scalar>,
        {
            type Output = Result<RegulatedRepr<Inner, Co, N>, Err>;

            fn execute(self, task: $repr<1, RegulatedRepr<A, Co, N>, E>) -> Self::Output {
                let (a, [axis_mapping], $op(scalar)) = task.into_raw();
                let (a, coeff) = a.into_raw();
                let (scalar, coeff_more) = N::scalar_regulate(scalar);
                let task = unsafe { $repr::from_raw_unchecked((a, [axis_mapping], $op(scalar))) };
                let repr = self.0.execute(task)?;
                Ok(RegulatedRepr {
                    repr,
                    coeff: coeff.$combine([coeff_more]),
                    _reg: PhantomData,
                })
            }
        }

        unsafe impl<Mk, C, A, E, Co, N, Inner, Err>
            ReprContext<Mk, 1, $repr<1, RegulatedRepr<A, Co, N>, E>> for RegulatedCtx<C>
        where
            A: TensorTupleRepr<1>,
            C: ReprContext<Mk, 1, $repr<1, A, E>, Repr = Inner, CType = Resulting<Raw, Err>>,
            Inner: TensorTupleRepr<1>,
            N: ScalarRegulator<E>,
            Co: CoefficientRepr<Scalar = N::Scalar>,
        {
            type Repr = RegulatedRepr<Inner, Co, N>;
            type CType = Resulting<Raw, Err>;
        }
    };
}

impl_regulated_scalar!(LeftScalarMulRepr, LeftScalarMulOp, mul);
impl_regulated_scalar!(RightScalarMulRepr, RightScalarMulOp, mul);
impl_regulated_scalar!(LeftScalarDivRepr, LeftScalarDivOp, div);
impl_regulated_scalar!(RightScalarDivRepr, RightScalarDivOp, div);