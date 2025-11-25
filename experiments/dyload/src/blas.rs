//! Bindings to [BLAS] \(Fortran).
//!
//! ## [Architecture]
//!
//! [architecture]: https://blas-lapack-rs.github.io/architecture
//! [blas]: https://en.wikipedia.org/wiki/BLAS

#![no_std]
use core::ffi::{c_char, c_double, c_float, c_int};

/// A complex number with 64-bit parts.
#[allow(bad_style)]
pub type c_double_complex = [c_double; 2];

/// A complex number with 32-bit parts.
#[allow(bad_style)]
pub type c_float_complex = [c_float; 2];

// Level 1
//
// http://www.netlib.org/blas/#_level_1

type SrotgFn = extern "C" fn(a: *mut c_float, b: *mut c_float, c: *mut c_float, s: *mut c_float);

// Single

type SrotmgFn = extern "C" fn(
    d1: *mut c_float,
    d2: *mut c_float,
    x1: *mut c_float,
    y1: *const c_float,
    param: *mut c_float,
);
type SrotFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
    y: *mut c_float,
    incy: *const c_int,
    c: *const c_float,
    s: *const c_float,
);
type SrotmFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
    y: *mut c_float,
    incy: *const c_int,
    param: *const c_float,
);
type SswapFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
    y: *mut c_float,
    incy: *const c_int,
);

type SscalFn =
    extern "C" fn(n: *const c_int, a: *const c_float, x: *mut c_float, incx: *const c_int);
type ScopyFn = extern "C" fn(
    n: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    y: *mut c_float,
    incy: *const c_int,
);

type SaxpyFn = extern "C" fn(
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    y: *mut c_float,
    incy: *const c_int,
);

type SdotFn = extern "C" fn(
    n: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
) -> c_float;

type SdsdotFn = extern "C" fn(
    n: *const c_int,
    sb: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
) -> c_float;

type Snrm2Fn = extern "C" fn(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
type Scnrm2Fn =
    extern "C" fn(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_float;
type SasumFn = extern "C" fn(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_float;
type IsamaxFn = extern "C" fn(n: *const c_int, x: *const c_float, incx: *const c_int) -> c_int;

// Double
type DrotgFn =
    extern "C" fn(a: *mut c_double, b: *mut c_double, c: *mut c_double, s: *mut c_double);
type DrotmgFn = extern "C" fn(
    d1: *mut c_double,
    d2: *mut c_double,
    x1: *mut c_double,
    y1: *const c_double,
    param: *mut c_double,
);
type DrotFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
    y: *mut c_double,
    incy: *const c_int,
    c: *const c_double,
    s: *const c_double,
);
type DrotmFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
    y: *mut c_double,
    incy: *const c_int,
    param: *const c_double,
);
type DswapFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
    y: *mut c_double,
    incy: *const c_int,
);
type DscalFn =
    extern "C" fn(n: *const c_int, a: *const c_double, x: *mut c_double, incx: *const c_int);
type DcopyFn = extern "C" fn(
    n: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    y: *mut c_double,
    incy: *const c_int,
);
type DaxpyFn = extern "C" fn(
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    y: *mut c_double,
    incy: *const c_int,
);
type DdotFn = extern "C" fn(
    n: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    y: *const c_double,
    incy: *const c_int,
) -> c_double;
type DsdotFn = extern "C" fn(
    n: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
) -> c_double;
type Dnrm2Fn = extern "C" fn(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_double;
type Dznrm2Fn =
    extern "C" fn(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_double;
type DasumFn = extern "C" fn(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_double;
type IdamaxFn = extern "C" fn(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_int;
// Complex
type CrotgFn = extern "C" fn(
    a: *mut c_float_complex,
    b: *const c_float_complex,
    c: *mut c_float,
    s: *mut c_float_complex,
);
type CsrotFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_float_complex,
    incx: *const c_int,
    y: *mut c_float_complex,
    incy: *const c_int,
    c: *const c_float,
    s: *const c_float,
);
type CswapFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_float_complex,
    incx: *const c_int,
    y: *mut c_float_complex,
    incy: *const c_int,
);
type CscalFn = extern "C" fn(
    n: *const c_int,
    a: *const c_float_complex,
    x: *mut c_float_complex,
    incx: *const c_int,
);
type CsscalFn =
    extern "C" fn(n: *const c_int, a: *const c_float, x: *mut c_float_complex, incx: *const c_int);
type CcopyFn = extern "C" fn(
    n: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *mut c_float_complex,
    incy: *const c_int,
);
type CaxpyFn = extern "C" fn(
    n: *const c_int,
    alpha: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *mut c_float_complex,
    incy: *const c_int,
);
type CdotuFn = extern "C" fn(
    pres: *mut c_float_complex,
    n: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
);
type CdotcFn = extern "C" fn(
    pres: *mut c_float_complex,
    n: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
);
type ScasumFn =
    extern "C" fn(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_float;
type IcamaxFn =
    extern "C" fn(n: *const c_int, x: *const c_float_complex, incx: *const c_int) -> c_int;

// Double complex
type ZrotgFn = extern "C" fn(
    a: *mut c_double_complex,
    b: *const c_double_complex,
    c: *mut c_double,
    s: *mut c_double_complex,
);
type ZdrotFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_double_complex,
    incx: *const c_int,
    y: *mut c_double_complex,
    incy: *const c_int,
    c: *const c_double,
    s: *const c_double,
);
type ZswapFn = extern "C" fn(
    n: *const c_int,
    x: *mut c_double_complex,
    incx: *const c_int,
    y: *mut c_double_complex,
    incy: *const c_int,
);
type ZscalFn = extern "C" fn(
    n: *const c_int,
    a: *const c_double_complex,
    x: *mut c_double_complex,
    incx: *const c_int,
);
type ZdscalFn = extern "C" fn(
    n: *const c_int,
    a: *const c_double,
    x: *mut c_double_complex,
    incx: *const c_int,
);
type ZcopyFn = extern "C" fn(
    n: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *mut c_double_complex,
    incy: *const c_int,
);

// Replaced extern fn declarations with pub type aliases (follow existing style)

// Level 1 continued (public)
pub type ZaxpyFn = extern "C" fn(
    n: *const c_int,
    alpha: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZdotuFn = extern "C" fn(
    pres: *mut c_double_complex,
    n: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
);
pub type ZdotcFn = extern "C" fn(
    pres: *mut c_double_complex,
    n: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
);
pub type DzasumFn =
    extern "C" fn(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_double;
pub type IzamaxFn =
    extern "C" fn(n: *const c_int, x: *const c_double_complex, incx: *const c_int) -> c_int;

// Level 2
//
// http://www.netlib.org/blas/#_level_2

// Single
pub type SgemvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);
pub type SgbmvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    kl: *const c_int,
    ku: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);
pub type SsymvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);
pub type SsbmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);
pub type SspmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    ap: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);
pub type StrmvFn = extern "C" fn(
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_float,
    lda: *const c_int,
    b: *mut c_float,
    incx: *const c_int,
);
pub type StbmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_float,
    lda: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
);
pub type StpmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_float,
    x: *mut c_float,
    incx: *const c_int,
);
pub type StrsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_float,
    lda: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
);
pub type StbsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_float,
    lda: *const c_int,
    x: *mut c_float,
    incx: *const c_int,
);
pub type StpsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_float,
    x: *mut c_float,
    incx: *const c_int,
);
pub type SgerFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
    a: *mut c_float,
    lda: *const c_int,
);
pub type SsyrFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    a: *mut c_float,
    lda: *const c_int,
);
pub type SsprFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    ap: *mut c_float,
);
pub type Ssyr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
    a: *mut c_float,
    lda: *const c_int,
);
pub type Sspr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
    ap: *mut c_float,
);

// Double
pub type DgemvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);
pub type DgbmvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    kl: *const c_int,
    ku: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);
pub type DsymvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);
pub type DsbmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);
pub type DspmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    ap: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);
pub type DtrmvFn = extern "C" fn(
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_double,
    lda: *const c_int,
    b: *mut c_double,
    incx: *const c_int,
);
pub type DtbmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_double,
    lda: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
);
pub type DtpmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_double,
    x: *mut c_double,
    incx: *const c_int,
);
pub type DtrsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_double,
    lda: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
);
pub type DtbsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_double,
    lda: *const c_int,
    x: *mut c_double,
    incx: *const c_int,
);
pub type DtpsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_double,
    x: *mut c_double,
    incx: *const c_int,
);
pub type DgerFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    y: *const c_double,
    incy: *const c_int,
    a: *mut c_double,
    lda: *const c_int,
);
pub type DsyrFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    a: *mut c_double,
    lda: *const c_int,
);
pub type DsprFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    ap: *mut c_double,
);
pub type Dsyr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    y: *const c_double,
    incy: *const c_int,
    a: *mut c_double,
    lda: *const c_int,
);
pub type Dspr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double,
    incx: *const c_int,
    y: *const c_double,
    incy: *const c_int,
    ap: *mut c_double,
);

// Complex
pub type CgemvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    beta: *const c_float_complex,
    y: *mut c_float_complex,
    incy: *const c_int,
);
pub type CgbmvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    kl: *const c_int,
    ku: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    beta: *const c_float_complex,
    y: *mut c_float_complex,
    incy: *const c_int,
);
pub type ChemvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    beta: *const c_float_complex,
    y: *mut c_float_complex,
    incy: *const c_int,
);
pub type ChbmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *const c_float_complex,
    incx: *const c_int,
    beta: *const c_float_complex,
    y: *mut c_float_complex,
    incy: *const c_int,
);
pub type ChpmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float_complex,
    ap: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    beta: *const c_float_complex,
    y: *mut c_float_complex,
    incy: *const c_int,
);
pub type CtrmvFn = extern "C" fn(
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *mut c_float_complex,
    incx: *const c_int,
);
pub type CtbmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *mut c_float_complex,
    incx: *const c_int,
);
pub type CtpmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_float_complex,
    x: *mut c_float_complex,
    incx: *const c_int,
);
pub type CtrsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *mut c_float_complex,
    incx: *const c_int,
);
pub type CtbsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_float_complex,
    lda: *const c_int,
    x: *mut c_float_complex,
    incx: *const c_int,
);
pub type CtpsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_float_complex,
    x: *mut c_float_complex,
    incx: *const c_int,
);
pub type CgeruFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
    a: *mut c_float_complex,
    lda: *const c_int,
);
pub type CgercFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
    a: *mut c_float_complex,
    lda: *const c_int,
);
pub type CherFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float_complex,
    incx: *const c_int,
    a: *mut c_float_complex,
    lda: *const c_int,
);
pub type ChprFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    x: *const c_float_complex,
    incx: *const c_int,
    ap: *mut c_float_complex,
);
pub type Chpr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
    ap: *mut c_float_complex,
);
pub type Cher2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float_complex,
    x: *const c_float_complex,
    incx: *const c_int,
    y: *const c_float_complex,
    incy: *const c_int,
    a: *mut c_float_complex,
    lda: *const c_int,
);

// Double complex
pub type ZgemvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    beta: *const c_double_complex,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZgbmvFn = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    kl: *const c_int,
    ku: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    beta: *const c_double_complex,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZhemvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    beta: *const c_double_complex,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZhbmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *const c_double_complex,
    incx: *const c_int,
    beta: *const c_double_complex,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZhpmvFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double_complex,
    ap: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    beta: *const c_double_complex,
    y: *mut c_double_complex,
    incy: *const c_int,
);
pub type ZtrmvFn = extern "C" fn(
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZtbmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZtpmvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_double_complex,
    x: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZtrsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZtbsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    k: *const c_int,
    a: *const c_double_complex,
    lda: *const c_int,
    x: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZtpsvFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const c_int,
    ap: *const c_double_complex,
    x: *mut c_double_complex,
    incx: *const c_int,
);
pub type ZgeruFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
    a: *mut c_double_complex,
    lda: *const c_int,
);
pub type ZgercFn = extern "C" fn(
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
    a: *mut c_double_complex,
    lda: *const c_int,
);
pub type ZherFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double_complex,
    incx: *const c_int,
    a: *mut c_double_complex,
    lda: *const c_int,
);
pub type ZhprFn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    x: *const c_double_complex,
    incx: *const c_int,
    ap: *mut c_double_complex,
);
pub type Zher2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
    a: *mut c_double_complex,
    lda: *const c_int,
);
pub type Zhpr2Fn = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double_complex,
    x: *const c_double_complex,
    incx: *const c_int,
    y: *const c_double_complex,
    incy: *const c_int,
    ap: *mut c_double_complex,
);

// Level 3
//
// http://www.netlib.org/blas/#_level_3

// Single
pub type SgemmFn = extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *const c_float,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);
pub type SsymmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *const c_float,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);
pub type SsyrkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);
pub type Ssyr2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *const c_float,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);
pub type StrmmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *mut c_float,
    ldb: *const c_int,
);
pub type StrsmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *mut c_float,
    ldb: *const c_int,
);

// Double
pub type DgemmFn = extern "C" fn(
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
pub type DsymmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *const c_double,
    ldb: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);
pub type DsyrkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);
pub type Dsyr2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
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
pub type DtrmmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *mut c_double,
    ldb: *const c_int,
);
pub type DtrsmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *mut c_double,
    ldb: *const c_int,
);

// Complex
pub type CgemmFn = extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *const c_float_complex,
    ldb: *const c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type CsymmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *const c_float_complex,
    ldb: *const c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type ChemmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *const c_float_complex,
    ldb: *const c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type CsyrkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type CherkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float_complex,
    lda: *const c_int,
    beta: *const c_float,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type Csyr2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *const c_float_complex,
    ldb: *const c_int,
    beta: *const c_float_complex,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type Cher2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *const c_float_complex,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float_complex,
    ldc: *const c_int,
);
pub type CtrmmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *mut c_float_complex,
    ldb: *const c_int,
);
pub type CtrsmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float_complex,
    a: *const c_float_complex,
    lda: *const c_int,
    b: *mut c_float_complex,
    ldb: *const c_int,
);

// Double complex
pub type ZgemmFn = extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *const c_double_complex,
    ldb: *const c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type ZsymmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *const c_double_complex,
    ldb: *const c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type ZhemmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *const c_double_complex,
    ldb: *const c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type ZsyrkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type ZherkFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double_complex,
    lda: *const c_int,
    beta: *const c_double,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type Zsyr2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *const c_double_complex,
    ldb: *const c_int,
    beta: *const c_double_complex,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type Zher2kFn = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *const c_double_complex,
    ldb: *const c_int,
    beta: *const c_double,
    c: *mut c_double_complex,
    ldc: *const c_int,
);
pub type ZtrmmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *mut c_double_complex,
    ldb: *const c_int,
);
pub type ZtrsmFn = extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double_complex,
    a: *const c_double_complex,
    lda: *const c_int,
    b: *mut c_double_complex,
    ldb: *const c_int,
);
