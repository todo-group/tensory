use tensory_ndarray::NdDenseTensor;
type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<Complex<f64>>;
fn main() -> anyhow::Result<()> {
    // Example usage of the negation operation
    let n = 10;

    let phys: Vec<_> = (0..n).map(|_| Leg::new()).collect();
    let psi_virt: Vec<_> = (0..n + 1).map(|_| Leg::new()).collect();
    let hamil_virt: Vec<_> = (0..n + 1).map(|_| Leg::new()).collect();

    let mut psi: Vec<_> = (0..n)
        .map(|i| Tensor::zero([(phys[i], 2), (psi_virt[i], 1), (psi_virt[i + 1], 1)]))
        .collect();
    let hamil: Vec<_> = (0..n)
        .map(|_| {
            Tensor::zero([
                (phys[i], 2),
                (phys[i].prime(), 2),
                (hamil_virt[i], 1),
                (hamil_virt[i + 1], 1),
            ])
        })
        .collect();

    let t: usize = 20;

    let blas: BlasRuntime = todo!(libload);

    for sweep in 0..t {
        for i in 0..(n - 1) {
            let left = Tensor::zero([]);
            for j in 0..(i - 1) {
                left = (left.view() * psi[j].view())?.with(&blas)?;
                left = (left.view() * hamil[j].view())?.with(&blas)?;
                left = (left.view()
                    * psi[j]
                        .view()
                        .conj()?
                        .with(())?
                        .replace(todo!("add prime to all legs")))?
                .with(&blas)?;
            }

            let right = Tensor::zero([]);
            for j in ((i + 2)..n).rev() {
                right = (right.view() * psi[j].view())?.with(&blas)?;
                right = (right.view() * hamil[j].view())?.with(&blas)?;
                right = (right.view()
                    * psi[j]
                        .view()
                        .conj()?
                        .with(())?
                        .replace(todo!("add prime to all legs")))?
                .with(&blas)?;
            }

            let center = (psi[i].view() * psi[i + 1].view())?.with(&blas)?;

            // optimizer using left,right,center,hamil[i],hamil[i+1]

            let (u, s, v) = center.view().svd()?.with(&blas)?;
            psi[i] = u;
            psi[i + 1] = (s.view() * v.view())?.with(&blas)?;
        }

        // do same as rev order
    }
}

fn test_simple_code() {
    let left = Tensor::zero([]);
    for j in 0..(i - 1) {
        left = (&left * &psi[j])? << &blas?; // idea: use ref as a suger of .view(). Ctx always consume tensor, but tensor could be ref repr.
        left = (&left * &hamil[j])? << &blas?;
        left = (&left
            * psi[j]
                .view()
                .conj()?
                .with(())?
                .replace(todo!("add prime to all legs")))?
            << &blas?;
        /* e.g. this chain seems easy to make mistake, but not.
        1. tensor contraction (A*B) is only defined for view * view. so the second operand must be a view.
        2. we cannot move psi[j], so we must take view.
        3. now we can consume view to genrated new view, which is internally memoed as conjugated.

        if delayed conj not implemented, view.conj() is not callable (precisely, no Ctx satisfy with(_)) condition).
        in this case psi[j].clone().conj()?.with(())? is the way, but this is apparently less efficient.
        */
    }

    let right = Tensor::zero([]);
    for j in ((i + 2)..n).rev() {
        right = (&right * &psi[j])? << &blas?;
        right = (&right * &hamil[j])? << &blas?;
        right = (&right
            * psi[j]
                .view()
                .conj()?
                .with(())?
                .replace(todo!("add prime to all legs")))?
            << &blas?;
    }

    let center = (&psi[i] * &psi[i + 1])? << &blas?;

    // optimizer using left,right,center,hamil[i],hamil[i+1]

    let (u, s, v) = &center.svd()? << &blas?;
    psi[i] = u;
    psi[i + 1] = (&s * &v)? << &blas?;
}
