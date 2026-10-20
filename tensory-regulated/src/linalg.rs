use core::marker::PhantomData;

use tensory_core::{
    concept::{
        container::{Raw, Resulting},
        task::{Context, IsTask},
    },
    repr::{ReprContext, TensorTupleRepr},
};
use tensory_linalg::svd::SvdRepr;

use crate::{CoefficientRepr, RegulatedCtx, RegulatedRepr, Regulator};

// commuttative scalar
// when A = U S V
// then a A = U (a S) V
impl<Mk, O, A, Co, N, U, S, V, Err> Context<
    Mk,
    SvdRepr<RegulatedRepr<A, Co, N>, ()>,
> for RegulatedCtx<O>
where
    A: TensorTupleRepr<1>,
    U: TensorTupleRepr<1>,
    S: TensorTupleRepr<1>,
    V: TensorTupleRepr<1>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
    N: Regulator<U> + Regulator<S> + Regulator<V>,
    (): ReprContext<
        Mk,
        3,
        SvdRepr<A, O>,
        Repr = (U, S, V),
        CType = Resulting<Raw, Err>,
    >,
{
    type Output = Result<
        (
            RegulatedRepr<<N as Regulator<U>>::Res, Co, N>,
            RegulatedRepr<<N as Regulator<S>>::Res, Co, N>,
            RegulatedRepr<<N as Regulator<V>>::Res, Co, N>,
        ),
        Err,
    >;

    fn execute(
        self,
        task: SvdRepr<RegulatedRepr<A, Co, N>, ()>,
    ) -> Self::Output {
        let (a, axes_split, ()) = task.into_raw();
        let (a, a_coeff) = a.into_raw();
        let task = unsafe { SvdRepr::from_raw_unchecked(a, axes_split, self.0) };
        let (u, s, v) = ().execute(task)?;
        let (u, u_coeff) = N::regulate(u);
        let (s, s_coeff) = N::regulate(s);
        let (v, v_coeff) = N::regulate(v);
        let u_coeff = Co::build(u_coeff);
        let s_coeff = a_coeff.mul([s_coeff]);
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

unsafe impl<Mk, O, A, Co, N, U, S, V, Err> ReprContext<
    Mk,
    3,
    SvdRepr<RegulatedRepr<A, Co, N>, ()>,
> for RegulatedCtx<O>
where
    A: TensorTupleRepr<1>,
    U: TensorTupleRepr<1>,
    S: TensorTupleRepr<1>,
    V: TensorTupleRepr<1>,
    Co: CoefficientRepr<Scalar = N::Scalar>,
    N: Regulator<U> + Regulator<S> + Regulator<V>,
    (): ReprContext<
        Mk,
        3,
        SvdRepr<A, O>,
        Repr = (U, S, V),
        CType = Resulting<Raw, Err>,
    >,
{
    type Repr = (
        RegulatedRepr<<N as Regulator<U>>::Res, Co, N>,
        RegulatedRepr<<N as Regulator<S>>::Res, Co, N>,
        RegulatedRepr<<N as Regulator<V>>::Res, Co, N>,
    );
    type CType = Resulting<Raw, Err>;
}
