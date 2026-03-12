use std::f64::consts::PI;
use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_linalg::prelude::*;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, NdRuntime};

#[allow(unused_imports)]
use rand::{SeedableRng, rngs::SmallRng};

// type aliases for convenience. You can change them to other implementations.
type Tensor<'a, E> = NdDenseTensor<E, VecMapper<&'a str>>;

fn main() -> anyhow::Result<()> {
    /*
       Important! bound tensor/runtime is developing layer. this example is just a snapshot of the current concept.
    */

    // in this example, we provide a usage of bound tensor.

    // again, we code a simplest example usage of tensory as follows:
    // t := <random tensor with legs a,b,c,d>
    // t_mul_pi := t * pi
    // t_mul_pi_div_pi := t_mul_pi / pi
    // t_diff := t_mul_pi_div_pi - t
    // print(norm(t_diff)/norm(t))

    // first we decides each sizes for each leg.
    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;

    // here we create a random tensor `t` with legs a,b,c,d.
    let t = Tensor::<f64>::random(lm!["a"=>a_n, "b"=>b_n, "c"=>c_n, "d"=>d_n])?;

    // here's the new point: we bind the tensor `t` to NdRuntime to obtain a bound tensor.
    // "bound" means that the tensor is associated with a specific runtime context for computation.
    let t = t.bind(NdRuntime);

    // here we perform scalar multiplication and division.
    let t_mul_pi = (t.clone() * PI)?; // ... (X) remember here we use t.clone()
    let t_mul_pi_div_pi = (t_mul_pi / PI)?;

    // as you can see, the the direct result of the `*` and `/` operation is already a bound tensor.
    // this is because `NdRuntime` internally provide the required context for these operations, so that the operation can be performed without explicit context passing.
    // so you don't need to call `.exec()` or `.with()` to perform the actual computation.
    // this api is high layer api, which is designed for ease of use.
    // in my opinion, this API is NOT recommended, because you cannot pass a custom context to the operation.
    // (Though we plan to add map-like API to allow users to pass non-default context, but it is still in design phase.)

    // the succeeding code is the same as before, we compute the difference and print the relative error.

    // here we perform subtraction to compute the difference between `t` and `t_mul_pi_div_pi`.
    let t_diff = (&t_mul_pi_div_pi - &t)?;

    // finally, we compute and print the norm of the difference, and normalize it by the norm of `t`.
    println!(
        "relative error (A ?= A * pi / pi): {}",
        (&t_diff).norm()? / (&t).norm()?
    );

    Ok(())
}
