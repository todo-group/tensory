use tensory_core::{
    arith::MulRepr,
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::{Context, RuntimeFor},
    },
    repr::ReprContext,
};

use crate::{
    Cuda, CudaDenseRepr, CudaDenseViewRepr, CudaErr, CudaRuntime, CudaScalar, contiguous_strides,
    is_contiguous,
};

impl<'l, 'r, E: CudaScalar>
    Context<Cuda, MulRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>> for CudaRuntime
{
    type Output = Result<CudaDenseRepr<E>, CudaErr>;

    fn execute(
        self,
        task: MulRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>,
    ) -> Self::Output {
        let (lhs, rhs, [connection], _) = task.into_raw();
        let ([lhs_dim, rhs_dim], pairs) = connection.into_raw();

        let mut lhs_conn = Vec::with_capacity(pairs.len());
        let mut rhs_conn = Vec::with_capacity(pairs.len());
        let mut lhs_used = vec![false; lhs_dim];
        let mut rhs_used = vec![false; rhs_dim];
        for &((t1, i1), (t2, i2)) in &pairs {
            let (l, r) = match (t1, t2) {
                (0, 1) => (i1, i2),
                (1, 0) => (i2, i1),
                _ => unreachable!("connection must join lhs and rhs"),
            };
            if lhs.shape[l] != rhs.shape[r] {
                return Err(CudaErr::InvalidInput);
            }
            lhs_conn.push(l);
            rhs_conn.push(r);
            lhs_used[l] = true;
            rhs_used[r] = true;
        }
        let lhs_free = (0..lhs_dim).filter(|&i| !lhs_used[i]);
        let rhs_free = (0..rhs_dim).filter(|&i| !rhs_used[i]);

        // Matrix layouts: lhs is (free, conn), rhs is (conn, free).
        let lhs_perm: Vec<usize> = lhs_free.clone().chain(lhs_conn).collect();
        let rhs_perm: Vec<usize> = rhs_conn.into_iter().chain(rhs_free.clone()).collect();
        let permute = |v: &[usize], perm: &[usize]| perm.iter().map(|&i| v[i]).collect::<Vec<_>>();
        let lhs_shape = permute(&lhs.shape, &lhs_perm);
        let rhs_shape = permute(&rhs.shape, &rhs_perm);
        let lhs_strides = permute(&lhs.strides, &lhs_perm);
        let rhs_strides = permute(&rhs.strides, &rhs_perm);

        let nconn = pairs.len();
        let m: usize = lhs_shape[..lhs_shape.len() - nconn].iter().product();
        let k: usize = lhs_shape[lhs_shape.len() - nconn..].iter().product();
        let n: usize = rhs_shape[nconn..].iter().product();

        let out_shape: Vec<usize> = lhs_free
            .map(|i| lhs.shape[i])
            .chain(rhs_free.map(|i| rhs.shape[i]))
            .collect();
        let out_strides = contiguous_strides(&out_shape);

        let mut out = self.stream().alloc_zeros::<E>(m * n)?;
        if m * n != 0 && k != 0 {
            let lhs_tmp = if is_contiguous(&lhs_shape, &lhs_strides) {
                None
            } else {
                Some(self.gather(lhs.data, &lhs_shape, &lhs_strides)?)
            };
            let rhs_tmp = if is_contiguous(&rhs_shape, &rhs_strides) {
                None
            } else {
                Some(self.gather(rhs.data, &rhs_shape, &rhs_strides)?)
            };
            E::gemm(
                &self.blas,
                m,
                n,
                k,
                lhs_tmp.as_ref().unwrap_or(lhs.data),
                rhs_tmp.as_ref().unwrap_or(rhs.data),
                &mut out,
            )?;
        }
        Ok(CudaDenseRepr::from_raw_layout(out, out_shape, out_strides))
    }
}

unsafe impl<'l, 'r, E: CudaScalar>
    ReprContext<Cuda, 1, MulRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>>
    for CudaRuntime
{
    type Repr = CudaDenseRepr<E>;
    type CType = Resulting<Raw, CudaErr>;
}

impl<'l, 'r, E: CudaScalar>
    RuntimeFor<MulRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>> for CudaRuntime
{
    type Mk = Cuda;
    type Ctx = CudaRuntime;
    fn ctx(&self) -> Self::Ctx {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use tensory_basic::{
        id::{Id128, Prime},
        mapper::VecMapper,
    };
    use tensory_core::prelude::*;

    use crate::{CudaDenseTensor, CudaDenseTensorExt, CudaRuntime};

    type Leg = Prime<Id128>;
    type Tensor = CudaDenseTensor<f32, VecMapper<Leg>>;

    // Requires a GPU.
    #[test]
    #[ignore]
    fn tensor_mul_test() -> Result<(), anyhow::Error> {
        let (a_n, b_n, i_n, j_n) = (2usize, 3usize, 4usize, 5usize);
        let (a, b, i, j) = (Leg::new(), Leg::new(), Leg::new(), Leg::new());

        // Different axis orders on both sides exercise the strided path.
        let ta_data: Vec<f32> = (0..a_n * i_n * j_n).map(|x| x as f32).collect();
        let tb_data: Vec<f32> = (0..j_n * b_n * i_n).map(|x| (x % 7) as f32).collect();

        let rt = CudaRuntime::new(0)?;
        let ta = Tensor::from_host(rt.stream(), lm![a => a_n, i => i_n, j => j_n], &ta_data)?;
        let tb = Tensor::from_host(rt.stream(), lm![j => j_n, b => b_n, i => i_n], &tb_data)?;

        let ta = ta.bind(rt.clone());
        let tb = tb.bind(rt);
        let tc = (&ta * &tb)?.unbind();

        let mut expected = vec![0f32; a_n * b_n];
        for ai in 0..a_n {
            for bi in 0..b_n {
                for ii in 0..i_n {
                    for ji in 0..j_n {
                        expected[ai * b_n + bi] += ta_data[(ai * i_n + ii) * j_n + ji]
                            * tb_data[(ji * b_n + bi) * i_n + ii];
                    }
                }
            }
        }
        assert_eq!(tc.repr().to_host()?, expected);
        Ok(())
    }
}
