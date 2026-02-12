use core::marker::PhantomData;

use tensory_core::{
    arith::{
        AddCtxImpl, CommutativeScalarMulCtx, LeftScalarMulCtx, MulCtxImpl, RightScalarMulCtx,
        SubCtxImpl,
    },
    mapper::{ConnectAxisOrigin, OverlayAxisMapping},
    repr::TensorRepr,
};

use crate::{CoefficientRepr, RegulatedCtx, RegulatedRepr, Regulation, Regulator, ScalarRegulator};

unsafe impl<
    C: MulCtxImpl<Lhs, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Regulator<C::Res>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> MulCtxImpl<RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>> for RegulatedCtx<C>
{
    type Res = RegulatedRepr<N::Res, Co, N>;
    type Err = C::Err;

    unsafe fn mul_unchecked(
        self,
        lhs: RegulatedRepr<Lhs, Co, N>,
        rhs: RegulatedRepr<Rhs, Co, N>,
        axis_origin: ConnectAxisOrigin<2>,
    ) -> Result<Self::Res, Self::Err> {
        // a A * b B = (a b) (A B) = (a b x) X
        let res = unsafe { self.0.mul_unchecked(lhs.repr, rhs.repr, axis_origin) }?;
        let coeff = Co::merge([lhs.coeff, rhs.coeff]);
        let (repr, coeff_more) = N::regulate(res);
        let coeff = coeff.mul([coeff_more]);
        Ok(RegulatedRepr {
            repr,
            coeff,
            _reg: PhantomData,
        })
    }
}

unsafe impl<
    C: WeightedAddCtxImpl<<N as Regulation>::Scalar, Lhs, <N as Regulation>::Scalar, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Regulator<C::Res>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> AddCtxImpl<RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>> for RegulatedCtx<C>
{
    type Res = RegulatedRepr<N::Res, Co, N>;
    type Err = C::Err;

    unsafe fn add_unchecked(
        self,
        lhs: RegulatedRepr<Lhs, Co, N>,
        rhs: RegulatedRepr<Rhs, Co, N>,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (coeff, [lhs_coeff, rhs_coeff]) = Co::factorize([lhs.coeff, rhs.coeff]);
        // let lhs_repr = N::scalar_mul(lhs.repr, lhs_coeff);
        // let rhs_repr = N::scalar_mul(rhs.repr, rhs_coeff);
        // let res = unsafe { self.0.add_unchecked(lhs_repr, rhs_repr, axis_mapping) }?;

        let res = unsafe {
            self.0
                .weighted_add_unchecked(lhs_coeff, lhs.repr, rhs_coeff, rhs.repr, axis_mapping)
        }?;

        let (res, coeff_more) = N::regulate(res);

        let coeff = coeff.mul([coeff_more]);

        Ok(RegulatedRepr {
            repr: res,
            coeff,
            _reg: PhantomData,
        })
    }
}

pub unsafe trait WeightedAddCtxImpl<LhsCo, Lhs: TensorRepr, RhsCo, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
    ///
    /// # Safety
    ///
    /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
    unsafe fn weighted_add_unchecked(
        self,
        lhs_coeff: LhsCo,
        lhs: Lhs,
        rhs_coeff: RhsCo,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}

// // hack of compiler
// pub unsafe trait WeightedAddCtxImplReordered<Lhs: TensorRepr, LhsCo, Rhs: TensorRepr, RhsCo> {
//     /// The type of the result tensor representation.
//     type Res: TensorRepr;
//     /// The type of the error returned by the context. (considered as internal error)
//     type Err;
//     /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
//     ///
//     /// # Safety
//     /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
//     unsafe fn weighted_add_unchecked_r(
//         self,
//         lhs_coeff: LhsCo,
//         lhs: Lhs,
//         rhs_coeff: RhsCo,
//         rhs: Rhs,
//         axis_mapping: OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err>;
// }

// unsafe impl<
//     T: WeightedAddCtxImplReordered<Lhs, LhsCo, Rhs, RhsCo>,
//     Lhs: TensorRepr,
//     LhsCo,
//     Rhs: TensorRepr,
//     RhsCo,
// > WeightedAddCtxImpl<LhsCo, Lhs, RhsCo, Rhs> for T
// {
//     type Res = T::Res;
//     type Err = T::Err;
//     unsafe fn weighted_add_unchecked(
//         self,
//         lhs_coeff: LhsCo,
//         lhs: Lhs,
//         rhs_coeff: RhsCo,
//         rhs: Rhs,
//         axis_mapping: OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         unsafe {
//             //T as WeightedAddCtxImplReordered
//             self.weighted_add_unchecked_r(lhs_coeff, lhs, rhs_coeff, rhs, axis_mapping)
//         }
//     }
// }

unsafe impl<
    C: WeightedSubCtxImpl<<N as Regulation>::Scalar, Lhs, <N as Regulation>::Scalar, Rhs>,
    Lhs: TensorRepr,
    Rhs: TensorRepr,
    N: Regulator<C::Res>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> SubCtxImpl<RegulatedRepr<Lhs, Co, N>, RegulatedRepr<Rhs, Co, N>> for RegulatedCtx<C>
{
    type Res = RegulatedRepr<N::Res, Co, N>;
    type Err = C::Err;

    unsafe fn sub_unchecked(
        self,
        lhs: RegulatedRepr<Lhs, Co, N>,
        rhs: RegulatedRepr<Rhs, Co, N>,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err> {
        let (coeff, [lhs_coeff, rhs_coeff]) = Co::factorize([lhs.coeff, rhs.coeff]);
        // let lhs_repr = N::scalar_mul(lhs.repr, lhs_coeff);
        // let rhs_repr = N::scalar_mul(rhs.repr, rhs_coeff);
        // let res = unsafe { self.0.add_unchecked(lhs_repr, rhs_repr, axis_mapping) }?;

        let res = unsafe {
            self.0
                .weighted_sub_unchecked(lhs_coeff, lhs.repr, rhs_coeff, rhs.repr, axis_mapping)
        }?;

        let (res, coeff_more) = N::regulate(res);

        let coeff = coeff.mul([coeff_more]);

        Ok(RegulatedRepr {
            repr: res,
            coeff,
            _reg: PhantomData,
        })
    }
}

pub unsafe trait WeightedSubCtxImpl<LhsCo, Lhs: TensorRepr, RhsCo, Rhs: TensorRepr> {
    /// The type of the result tensor representation.
    type Res: TensorRepr;
    /// The type of the error returned by the context. (considered as internal error)
    type Err;

    /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
    ///
    /// # Safety
    ///
    /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
    unsafe fn weighted_sub_unchecked(
        self,
        lhs_coeff: LhsCo,
        lhs: Lhs,
        rhs_coeff: RhsCo,
        rhs: Rhs,
        axis_mapping: OverlayAxisMapping<2>,
    ) -> Result<Self::Res, Self::Err>;
}

// // hack of compiler
// pub unsafe trait WeightedSubCtxImplReordered<Lhs: TensorRepr, LhsCo, Rhs: TensorRepr, RhsCo> {
//     /// The type of the result tensor representation.
//     type Res: TensorRepr;
//     /// The type of the error returned by the context. (considered as internal error)
//     type Err;

//     /// Performs addition operation on the tensors `lhs` and `rhs` with the given axis overlay mapping.
//     ///
//     /// # Safety
//     ///
//     /// the user MUST ensure that `axis_mapping` has the same number of axes same as the input tensors.
//     unsafe fn weighted_sub_unchecked(
//         self,
//         lhs_coeff: LhsCo,
//         lhs: Lhs,
//         rhs_coeff: RhsCo,
//         rhs: Rhs,
//         axis_mapping: OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err>;
// }

// unsafe impl<
//     T: WeightedSubCtxImplReordered<Lhs, LhsCo, Rhs, RhsCo>,
//     Lhs: TensorRepr,
//     LhsCo,
//     Rhs: TensorRepr,
//     RhsCo,
// > WeightedSubCtxImpl<LhsCo, Lhs, RhsCo, Rhs> for T
// {
//     type Res = T::Res;
//     type Err = T::Err;

//     unsafe fn weighted_sub_unchecked(
//         self,
//         lhs_coeff: LhsCo,
//         lhs: Lhs,
//         rhs_coeff: RhsCo,
//         rhs: Rhs,
//         axis_mapping: OverlayAxisMapping<2>,
//     ) -> Result<Self::Res, Self::Err> {
//         unsafe { self.weighted_sub_unchecked(lhs_coeff, lhs, rhs_coeff, rhs, axis_mapping) }
//     }
// }

// pub unsafe trait LeftMulRegulator<A: TensorRepr, E>: Regulation {
//     type Res: TensorRepr;
//     fn left_mul_regulate(scalar: E, repr: A) -> (Self::Res, Self::Scalar);
// }
// pub unsafe trait RightMulRegulator<A: TensorRepr, E>: Regulation {
//     type Res: TensorRepr;
//     fn right_mul_regulate(repr: A, scalar: E) -> (Self::Res, Self::Scalar);
// }

unsafe impl<
    C: LeftScalarMulCtx<A, E>,
    A: TensorRepr,
    E,
    N: ScalarRegulator<E>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> LeftScalarMulCtx<RegulatedRepr<A, Co, N>, E> for RegulatedCtx<C>
{
    type Res = RegulatedRepr<C::Res, Co, N>;

    type Err = C::Err;

    fn left_scalar_mul(
        self,
        a: RegulatedRepr<A, Co, N>,
        scalar: E,
    ) -> Result<Self::Res, Self::Err> {
        let (phase, coeff_more) = N::scalar_regulate(scalar);
        let (a, coeff) = a.into_raw();
        let res = self.0.left_scalar_mul(a, phase)?;
        let coeff = coeff.mul([coeff_more]);
        Ok(RegulatedRepr {
            repr: res,
            coeff,
            _reg: PhantomData,
        })
    }
}

unsafe impl<
    C: RightScalarMulCtx<A, E>,
    A: TensorRepr,
    E,
    N: ScalarRegulator<E>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> RightScalarMulCtx<RegulatedRepr<A, Co, N>, E> for RegulatedCtx<C>
{
    type Res = RegulatedRepr<C::Res, Co, N>;

    type Err = C::Err;

    fn right_scalar_mul(
        self,
        a: RegulatedRepr<A, Co, N>,
        scalar: E,
    ) -> Result<Self::Res, Self::Err> {
        let (phase, coeff_more) = N::scalar_regulate(scalar);
        let (a, coeff) = a.into_raw();
        let res = self.0.right_scalar_mul(a, phase)?;
        let coeff = coeff.mul([coeff_more]);
        Ok(RegulatedRepr {
            repr: res,
            coeff,
            _reg: PhantomData,
        })
    }
}
