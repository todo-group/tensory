use std::f64::consts::PI;
use tensory_basic::mapper::VecMapper;
use tensory_core::prelude::*;
use tensory_linalg::norm::TensorNormExt;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt};

#[allow(unused_imports)]
use rand::{SeedableRng, rngs::SmallRng};

// type aliases for convenience. You can change them to other implementations.
type Tensor<E> = NdDenseTensor<E, VecMapper<&'static str>>;

fn main() -> anyhow::Result<()> {
    // in this example, we provide a simplest usage of tensory for understanding the basic schema of tensory operations.

    // we code a simplest example usage of tensory as follows:
    // a:= <random tensor with legs a,b,c,d>
    // b:= a * pi
    // c:= b / pi
    // d:= c - a
    // print(norm(d)/norm(a))

    // first we decides each sizes for each leg.
    let a_n = 10;
    let b_n = 20;
    let c_n = 30;
    let d_n = 40;

    // here we create a random tensor `t` with legs a,b,c,d.
    let t = Tensor::<f64>::random(lm!["a"=>a_n, "b"=>b_n, "c"=>c_n, "d"=>d_n])?;

    // `lm!` is a macro to build leg-maps: some value is bound with leg ID.
    // here we pass leg IDs as &'static str and their sizes as usize.
    // so this constructor builds a random tensor with legs "a", "b", "c", "d" and their sizes a_n, b_n, c_n, d_n respectively.

    // alternatively, you can use `Tensor::random_using` to use fast and seedable random number generator (RNG) for speed and reproducibility. you will notice the slowness of the default RNG by commenting out the line above and uncommenting the following lines.
    // let mut rng = SmallRng::seed_from_u64(0);
    // let t = Tensor::<f64>::random_using(lm!["a"=>a_n, "b"=>b_n, "c"=>c_n, "d"=>d_n], &mut rng)?;

    // here we perform scalar multiplication and division.
    let t_mul_pi = (t.clone() * PI).with(())?; // ... (X) remember here we use t.clone()
    let t_mul_pi_div_pi = (t_mul_pi / PI).exec()?;

    // as you can see, the the direct result of the `*` and `/` operation is not a tensor, but a middle task struct.
    // in this case, it is TensorRightScalarMul and TensorRightScalarDiv respectively.
    // the task building is lazy. the actual computation is performed when we call `.with(ctx)` method.
    // `ctx` implements the required context trait for the operation. so you can switch different contexts to change the computation strategy.
    // also, `ctx` has a role to pass a required resource, like memory allocator or CUBLAS handle.
    // in these scalar operations, the context is `()`, which means no special context is required.
    // they are considered default context. you can use `.exec()` method instead as a syntax sugar, as the second line.

    // here we perform subtraction to compute the difference between `t` and `t_mul_pi_div_pi`.
    let t_diff = (&t_mul_pi_div_pi - &t)?.exec()?;

    // you will notice that we use `&t` and `&t_mul_pi_div_pi` here, instead of `t` and `t_mul_pi_div_pi`.
    // actually, this is a syntax suger for:
    // let t_diff = (t_mul_pi_div_pi.view() - t.view())?.exec()?;
    // `.view()` method creates a view tensor that just borrows the data of the original tensor.
    // in this case, they create NdDenseViewTensor.
    // tensory-ndarary provides subtraction operation only for view tensors, since the subtraction does not consume the tensor.
    // of course, this is the implementation of tensory-ndarray. other third-party tensor crates may consume the owned tensor by subtraction.
    // so the internal trait definition is declared to consume the input tensors.
    // therefore we provide `.view()` as a workaround to create tensor references.
    // and, to avoid to write `.view()` every time, we overload the `-` operator for references to tensors.

    // finally, we compute and print the norm of the difference, and normalize it by the norm of `t`.
    println!(
        "relative error (A ?= A * pi / pi): {}",
        (&t_diff).norm().exec()? / (&t).norm().exec()?
    );

    // again, we use `&t_diff` and `&t` to borrow the tensors for norm computation.
    // here we use `t` again, so if we had not used `t.clone()` before, `t` would have been moved and this line would cause a compile error. (see the comment with (X) mark above)
    // also, you must use paranthesis as `(&t_diff)` and `(&t)` here, because the method call has higher precedence than the reference operator `&`.
    // if we had not used `&` here, the compiler would think like this:
    // `&t.norm().exec()` = `&( ( t.norm() ).exec() )`
    // which means `t.norm()` consumes `t`, and then call the actual computation from default context `()`, but tensory-ndarray does not implement norm operation for owned tensors, only for view tensors.
    // so the compiler would raise an error that `.exec()` is not satisfied trait bound.

    // that's all for this simple example.
    // now you should understand the basic schema of tensory operations.

    Ok(())
}
