use num_complex::Complex;
use tensory_basic::{
    id::{Id128, Prime},
    mapper::VecMapper,
};
use tensory_core::leg;
use tensory_core::repr::TensorRepr;
use tensory_linalg::svd::TensorSvdExt;
use tensory_ndarray::{NdDenseTensor, NdDenseTensorExt, cut_filter::MaxIx};

type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<f64, VecMapper<Leg>>;
fn main() -> anyhow::Result<()> {
    // Example usage of the negation operation
    let t: usize = 20;
    let d: usize = 10;

    let mut x = Leg::new();
    let mut y = Leg::new();

    let mut z = Tensor::random(leg![x=> 2, y=> 2, x.prime()=> 2, y.prime()=> 2]).unwrap();
    for x_i in 0..2 {
        for y_i in 0..2 {
            for x_p_i in 0..2 {
                for y_p_i in 0..2 {
                    z[leg![
                        &x=>x_i,
                        &y=>y_i,
                        &x.prime()=>x_p_i,
                        &y.prime()=>y_p_i
                    ]] = 0.0;
                }
            }
        }
    }

    for _renorm in 0..t {
        std::println!("order {}", z.repr().naxes());

        let x_new = Leg::new();
        let y_new = Leg::new();

        let dum = Leg::new();

        let (u, s, v) = (z
            .view()
            .replace_leg(leg![&x => x.prime(),&y => y.prime(), &x.prime() => x, &y.prime() => y])
            .unwrap())
        .svd_with_more_ids(leg![&x, &y], x_new, x_new.prime(), dum, dum)?
        .with((MaxIx(d),))?;
        let A = u;
        let B = (&s * &v)?.with(())?;

        let (u, s, v) = (&z)
            .svd_with_more_ids(leg![&x, &y.prime()], y_new, y_new.prime(), dum, dum)?
            .with((MaxIx(d),))?;

        let C = u;
        let D = (&s * &v)?.with(())?;

        // bound tensor should provide common single runtime const reference
        // and options are passed in first call fn
        // then convert it to the pair of runtime and option
        // this tuple should impl XxxCtx

        z = (&(&A * &D)?.with(())? * &(&B * &C)?.with(())?)?.with(())?;

        x = x_new;
        y = y_new;
    }
    Ok(())
}
