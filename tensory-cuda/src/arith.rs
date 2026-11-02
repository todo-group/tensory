use tensory_core::{
    arith::{AddRepr, MulRepr, SubRepr},
    concept::{
        container::{Raw, Resulting},
        refined::RefinedFrom,
        task::{Context, RuntimeFor},
    },
    op::AxisAllocation,
    repr::ReprContext,
};

use crate::{
    Cuda, CudaDenseRepr, CudaDenseViewRepr, CudaErr, CudaRuntime, CudaScalar, cutensor::Operand,
};

type Mul<'l, 'r, E> = MulRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>;
type Add<'l, 'r, E> = AddRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>;
type Sub<'l, 'r, E> = SubRepr<1, CudaDenseViewRepr<'l, E>, CudaDenseViewRepr<'r, E>>;

impl<'l, 'r, E: CudaScalar> Context<Cuda, Mul<'l, 'r, E>> for CudaRuntime {
    type Output = Result<CudaDenseRepr<E>, CudaErr>;

    fn execute(self, task: Mul<'l, 'r, E>) -> Self::Output {
        let (lhs, rhs, [connection], _) = task.into_raw();
        let ([lhs_dim, rhs_dim], pairs) = connection.into_raw();

        // Each axis gets a unique mode label; connected axes share the lhs label.
        let lhs_modes: Vec<i32> = (0..lhs_dim as i32).collect();
        let mut rhs_modes: Vec<i32> = (lhs_dim as i32..(lhs_dim + rhs_dim) as i32).collect();
        for &((t1, i1), (t2, i2)) in &pairs {
            let (l, r) = match (t1, t2) {
                (0, 1) => (i1, i2),
                (1, 0) => (i2, i1),
                _ => unreachable!("connection must join lhs and rhs"),
            };
            if lhs.shape[l] != rhs.shape[r] {
                return Err(CudaErr::InvalidInput);
            }
            rhs_modes[r] = lhs_modes[l];
        }

        let mut out_shape = Vec::new();
        let mut out_modes = Vec::new();
        for (i, &m) in lhs_modes.iter().enumerate() {
            if !rhs_modes.contains(&m) {
                out_shape.push(lhs.shape[i]);
                out_modes.push(m);
            }
        }
        for (i, &m) in rhs_modes.iter().enumerate() {
            if !lhs_modes.contains(&m) {
                out_shape.push(rhs.shape[i]);
                out_modes.push(m);
            }
        }

        self.contract(
            Operand {
                data: lhs.data,
                shape: &lhs.shape,
                strides: &lhs.strides,
                modes: lhs_modes,
            },
            Operand {
                data: rhs.data,
                shape: &rhs.shape,
                strides: &rhs.strides,
                modes: rhs_modes,
            },
            out_shape,
            &out_modes,
        )
    }
}

/// Computes `lhs + gamma * rhs`, permuting both operands onto the result axes.
fn add_scaled<E: CudaScalar>(
    rt: &CudaRuntime,
    lhs: CudaDenseViewRepr<'_, E>,
    rhs: CudaDenseViewRepr<'_, E>,
    mapping: AxisAllocation<2>,
    rhs_coeff: E,
) -> Result<CudaDenseRepr<E>, CudaErr> {
    let (_, [lhs_perm, rhs_perm]) = mapping.into_raw();

    // Result axis j is axis `perm[j]` of the input, so that input axis gets label j.
    let modes_of = |perm: &[usize]| {
        let mut modes = vec![0i32; perm.len()];
        for (j, &p) in perm.iter().enumerate() {
            modes[p] = j as i32;
        }
        modes
    };
    let out_shape: Vec<usize> = lhs_perm.iter().map(|&p| lhs.shape[p]).collect();
    let rhs_shape: Vec<usize> = rhs_perm.iter().map(|&p| rhs.shape[p]).collect();
    if out_shape != rhs_shape {
        return Err(CudaErr::InvalidInput);
    }

    // The output keeps lhs's axis order; the result order is a stride permutation.
    let out = rt.add_scaled(
        Operand {
            data: rhs.data,
            shape: &rhs.shape,
            strides: &rhs.strides,
            modes: modes_of(&rhs_perm),
        },
        rhs_coeff,
        Operand {
            data: lhs.data,
            shape: &lhs.shape,
            strides: &lhs.strides,
            modes: modes_of(&lhs_perm),
        },
        E::ONE,
    )?;
    let (data, shape, strides) = out.into_raw();
    Ok(CudaDenseRepr::from_raw_layout(
        data,
        lhs_perm.iter().map(|&p| shape[p]).collect(),
        lhs_perm.iter().map(|&p| strides[p]).collect(),
    ))
}

impl<'l, 'r, E: CudaScalar> Context<Cuda, Add<'l, 'r, E>> for CudaRuntime {
    type Output = Result<CudaDenseRepr<E>, CudaErr>;

    fn execute(self, task: Add<'l, 'r, E>) -> Self::Output {
        let (lhs, rhs, [mapping], _) = task.into_raw();
        add_scaled(&self, lhs, rhs, mapping, E::ONE)
    }
}

impl<'l, 'r, E: CudaScalar> Context<Cuda, Sub<'l, 'r, E>> for CudaRuntime {
    type Output = Result<CudaDenseRepr<E>, CudaErr>;

    fn execute(self, task: Sub<'l, 'r, E>) -> Self::Output {
        let (lhs, rhs, [mapping], _) = task.into_raw();
        add_scaled(&self, lhs, rhs, mapping, -E::ONE)
    }
}

macro_rules! impl_runtime {
    ($($task:ident),*) => {$(
        unsafe impl<'l, 'r, E: CudaScalar> ReprContext<Cuda, 1, $task<'l, 'r, E>> for CudaRuntime {
            type Repr = CudaDenseRepr<E>;
            type CType = Resulting<Raw, CudaErr>;
        }

        impl<'l, 'r, E: CudaScalar> RuntimeFor<$task<'l, 'r, E>> for CudaRuntime {
            type Mk = Cuda;
            type Ctx = CudaRuntime;
            fn ctx(&self) -> Self::Ctx {
                self.clone()
            }
        }
    )*};
}
impl_runtime!(Mul, Add, Sub);

#[cfg(test)]
mod tests {
    use tensory_basic::{
        id::{Id128, Prime},
        mapper::VecMapper,
    };
    use tensory_core::prelude::*;

    use crate::{CudaDenseRepr, CudaDenseTensor, CudaDenseTensorExt, CudaRuntime};

    type Leg = Prime<Id128>;
    type Tensor = CudaDenseTensor<f32, VecMapper<Leg>>;

    // Requires a GPU and cuTENSOR.
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

        let tc = (&ta * &tb)?.with(rt)?;

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

    // Requires a GPU and cuTENSOR.
    #[test]
    #[ignore]
    fn tensor_add_sub_test() -> Result<(), anyhow::Error> {
        let (a_n, b_n, c_n) = (2usize, 3usize, 4usize);
        let (a, b, c) = (Leg::new(), Leg::new(), Leg::new());

        let ta_data: Vec<f32> = (0..a_n * b_n * c_n).map(|x| x as f32).collect();
        let tb_data: Vec<f32> = (0..c_n * a_n * b_n).map(|x| (x * x % 11) as f32).collect();

        let rt = CudaRuntime::new(0)?;
        let ta = Tensor::from_host(rt.stream(), lm![a => a_n, b => b_n, c => c_n], &ta_data)?;
        // Same legs in a different axis order.
        let tb = Tensor::from_host(rt.stream(), lm![c => c_n, a => a_n, b => b_n], &tb_data)?;

        let sum = (&ta + &tb)?.with(rt.clone())?;
        let diff = (&ta - &tb)?.with(rt)?;

        let at = |ai: usize, bi: usize, ci: usize| {
            (
                ta_data[(ai * b_n + bi) * c_n + ci],
                tb_data[(ci * a_n + ai) * b_n + bi],
            )
        };
        let mut sum_expected = Vec::new();
        let mut diff_expected = Vec::new();
        // Result axes follow the lhs order (a, b, c).
        for ai in 0..a_n {
            for bi in 0..b_n {
                for ci in 0..c_n {
                    let (x, y) = at(ai, bi, ci);
                    sum_expected.push(x + y);
                    diff_expected.push(x - y);
                }
            }
        }
        assert_eq!(sum.repr().to_host()?, sum_expected);
        assert_eq!(diff.repr().to_host()?, diff_expected);
        Ok(())
    }

    // Requires a GPU and cuTENSOR.
    #[test]
    #[ignore]
    fn tensor_add_keeps_lhs_memory_order() -> Result<(), anyhow::Error> {
        let (a_n, b_n) = (2usize, 3usize);
        let (a, b) = (Leg::new(), Leg::new());

        // lhs is a column-major (a, b) matrix; rhs is row-major.
        let lhs_data: Vec<f32> = (0..a_n * b_n).map(|x| x as f32).collect(); // [b][a]
        let rhs_data: Vec<f32> = (0..a_n * b_n).map(|x| (10 * x) as f32).collect(); // [a][b]

        let rt = CudaRuntime::new(0)?;
        let rhs = Tensor::from_host(rt.stream(), lm![a => a_n, b => b_n], &rhs_data)?;
        let lhs_repr = CudaDenseRepr::from_raw_layout(
            rt.stream().clone_htod(&lhs_data)?,
            vec![a_n, b_n],
            vec![1, a_n],
        );
        let mapper = rhs.mapper_array()[0].clone();
        let lhs = unsafe { Tensor::from_raw_unchecked(lhs_repr, [mapper]) };

        let sum = (&lhs + &rhs)?.with(rt)?;

        assert_eq!(sum.repr().strides(), [1, a_n]);
        let mut expected = Vec::new();
        for ai in 0..a_n {
            for bi in 0..b_n {
                expected.push(lhs_data[bi * a_n + ai] + rhs_data[ai * b_n + bi]);
            }
        }
        assert_eq!(sum.repr().to_host()?, expected);
        Ok(())
    }
}
