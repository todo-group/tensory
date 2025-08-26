use tensory_ndarray::NdDenseTensor;

type Leg = Prime<Id128>;
type Tensor = NdDenseTensor<Complex<f64>>;
fn main() {
    // Example usage of the negation operation
    let t: usize = 20;
    let d: usize = 10;

    let mut x = Leg::new();
    let mut y = Leg::new();
    let z = Tensor::zero([(x, 2), (y, 2), (x.prime(), 2), (y.prime(), 2)]);
    for x_i in 0..2 {
        for y_i in 0..2 {
            for x_p_i in 0..2 {
                for y_p_i in 0..2 {
                    z[[
                        (&x, x_i),
                        (&y, y_i),
                        (&x.prime(), y_p_i),
                        (&y.prime(), y_p_i),
                    ]] = Complex::new(0.0, 0.0);
                }
            }
        }
    }

    let blas;

    let d: usize = 10;

    for renorm in 0..t {
        let z;

        let x_new = Leg::new();
        let y_new = Leg::new();

        let (A, B) = z
            .view()
            .replace_id([
                (&x, x.prime()),
                (&y, y.prime()),
                (&x.prime(), x),
                (&y.prime(), y),
            ])
            .split_with_more_id([&x, &y], x_new, x_new.prime())
            << (&blas, MaxDim(d));
        let (C, D) =
            &z.split_with_more_id([&x, &y.prime()], y_new, y_new.prime()) << (&blas, MaxDim(d));

        let z = (&((&A * &D)? << &blas?) * &((&B * &C)? << &blas?)) << &blas?;

        x = x_new;
        y = y_new;
    }
}
