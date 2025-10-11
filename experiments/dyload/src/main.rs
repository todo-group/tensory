use anyhow::anyhow;
use std::ffi::{CStr, c_char, c_double, c_int};

// from openblas-sys
type DgemmFn = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *const c_double,
    ldb: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);

type GetConfigFn = unsafe extern "C" fn() -> *const c_char;

fn main() -> anyhow::Result<()> {
    let m: usize = 10;
    let k: usize = 30;
    let n: usize = 20;

    let mut a = vec![0.0; m * k];
    let mut b = vec![0.0; k * n];

    for i in 0..m {
        for j in 0..k {
            a[i * k + j] = 2.0 * i as f64 - 3.0 * j as f64;
        }
    }

    for i in 0..k {
        for j in 0..n {
            b[i * n + j] = 5.0 * i as f64 + 4.0 * j as f64;
        }
    }
    let a = a;
    let b = b;

    let c_hand = {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    };

    #[cfg(target_os = "linux")]
    const LIB_FILE_NAME: &str = "libopenblas.so";
    #[cfg(target_os = "macos")]
    const LIB_FILE_NAME: &str = "libopenblas.dylib";
    #[cfg(target_os = "windows")]
    const LIB_FILE_NAME: &str = "openblas.dll";

    let c_blas = unsafe {
        let mut c = vec![0.0; m * n];

        let lib_info = pkg_config::Config::new()
            .cargo_metadata(false)
            .env_metadata(false)
            .probe("openblas")?;
        let lib_path = lib_info
            .link_paths
            .into_iter()
            .map(|p| p.join(LIB_FILE_NAME))
            .find(|p| p.exists())
            .ok_or(anyhow!("no lib file detected"))?;

        let lib = libloading::Library::new(lib_path)?;
        let dgemm: libloading::Symbol<DgemmFn> = lib.get(b"dgemm_")?;

        let get_config: libloading::Symbol<GetConfigFn> = lib.get(b"openblas_get_config")?;
        // plan: reading the config string, the blas runtime creation may fail (prevent unsound runtime generation)
        // e.g. BlasRuntime<generics>(lib)?;

        let x = CStr::from_ptr(get_config());
        println!("OpenBLAS config: {}", x.to_str()?);

        // for fortran, a as k * m, b as n * k, c as n * m

        let m = m as c_int;
        let k = k as c_int;
        let n = n as c_int;

        let alpha = 1.0;
        let beta = 0.0;

        let transa = b"N" as *const u8 as *const c_char;
        let transb = b"N" as *const u8 as *const c_char;

        dgemm(
            transa,
            transb,
            &n,
            &m,
            &k,
            &alpha,
            b.as_ptr(),
            &n,
            a.as_ptr(),
            &k,
            &beta,
            c.as_mut_ptr(),
            &n,
        );
        c
    };

    let eps = 1e-10;

    for i in 0..m {
        for j in 0..n {
            assert!((c_hand[i * n + j] - c_blas[i * n + j]).abs() < eps);
        }
    }

    Ok(())
}
