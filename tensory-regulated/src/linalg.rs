use core::marker::PhantomData;

use tensory_core::repr::TensorRepr;
use tensory_linalg::svd::SvdCtxImpl;

use crate::{CoefficientRepr, RegulatedCtx, RegulatedRepr, Regulator};

// commuttative scalar
// when A = U S V
// then a A = U (a S) V
unsafe impl<
    C: SvdCtxImpl<A>,
    A: TensorRepr,
    N: Regulator<C::U> + Regulator<C::S> + Regulator<C::V>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
> SvdCtxImpl<RegulatedRepr<A, Co, N>> for RegulatedCtx<C>
{
    type U = RegulatedRepr<<N as Regulator<C::U>>::Res, Co, N>;

    type S = RegulatedRepr<<N as Regulator<C::S>>::Res, Co, N>;

    type V = RegulatedRepr<<N as Regulator<C::V>>::Res, Co, N>;
    type Err = C::Err;

    unsafe fn svd_unchecked(
        self,
        a: RegulatedRepr<A, Co, N>,
        axes_split: tensory_core::prelude::GroupedAxes<2>,
    ) -> Result<(Self::U, Self::S, Self::V), Self::Err> {
        let (u, s, v) = unsafe { self.0.svd_unchecked(a.repr, axes_split) }?;
        let (u, u_coeff) = N::regulate(u);
        let (s, s_coeff) = N::regulate(s);
        let (v, v_coeff) = N::regulate(v);
        let u_coeff = Co::build(u_coeff);
        let s_coeff = a.coeff.mul([s_coeff]);
        let v_coeff = Co::build(v_coeff);
        Ok((
            RegulatedRepr {
                repr: u,
                coeff: u_coeff,
                _reg: PhantomData,
            },
            RegulatedRepr {
                repr: s,
                coeff: s_coeff,
                _reg: PhantomData,
            },
            RegulatedRepr {
                repr: v,
                coeff: v_coeff,
                _reg: PhantomData,
            },
        ))
    }
}
