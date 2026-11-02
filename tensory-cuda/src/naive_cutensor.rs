//! Raw cuTENSOR elementwise-binary calls, to isolate cuTENSOR behavior from the abstractions.

use std::{ffi::c_void, mem::MaybeUninit};

use cudarc::{
    cutensor::{result, sys},
    driver::{CudaContext, DevicePtr, DevicePtrMut},
};

use crate::CudaScalar;

/// Tensor layout: extents, strides (in elements) and mode labels.
struct Layout<'a> {
    shape: &'a [i64],
    strides: &'a [i64],
    modes: &'a [i32],
}

fn step(name: &str, status: sys::cutensorStatus_t) -> Result<(), String> {
    match status {
        sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS => Ok(()),
        s => Err(format!("{name}: {s:?}")),
    }
}

fn desc(
    h: sys::cutensorHandle_t,
    name: &str,
    l: &Layout,
) -> Result<sys::cutensorTensorDescriptor_t, String> {
    unsafe {
        result::create_tensor_descriptor(
            h,
            l.shape.len() as u32,
            l.shape.as_ptr(),
            l.strides.as_ptr(),
            f32::DATA_TYPE,
            128,
        )
    }
    .map_err(|e| format!("create_tensor_descriptor({name}): {e:?}"))
}

/// D = a + c; `d_desc_of_c` makes D reuse C's descriptor instead of creating a separate one.
fn run(
    a: (&Layout, &[f32]),
    c: (&Layout, &[f32]),
    d: &Layout,
    d_desc_of_c: bool,
) -> Result<Vec<f32>, String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
    let stream = ctx.default_stream();
    let e = |e: cudarc::driver::DriverError| format!("{e:?}");
    ctx.bind_to_thread().map_err(e)?;

    let h = result::create_handle().map_err(|e| format!("create_handle: {e:?}"))?;
    let desc_a = desc(h, "A", a.0)?;
    let desc_c = desc(h, "C", c.0)?;
    let desc_d = if d_desc_of_c {
        desc_c
    } else {
        desc(h, "D", d)?
    };

    let identity = sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY;
    let mut op = MaybeUninit::uninit();
    step("create_elementwise_binary", unsafe {
        sys::cutensorCreateElementwiseBinary(
            h,
            op.as_mut_ptr(),
            desc_a,
            a.0.modes.as_ptr(),
            identity,
            desc_c,
            c.0.modes.as_ptr(),
            identity,
            desc_d,
            d.modes.as_ptr(),
            sys::cutensorOperator_t::CUTENSOR_OP_ADD,
            f32::compute_desc(),
        )
    })?;
    let op = unsafe { op.assume_init() };

    let pref = unsafe {
        result::create_plan_preference(
            h,
            sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
            sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
        )
    }
    .map_err(|e| format!("create_plan_preference: {e:?}"))?;
    let plan = unsafe { result::create_plan(h, op, pref, 0) }
        .map_err(|e| format!("create_plan: {e:?}"))?;

    let a_dev = stream.clone_htod(a.1).map_err(e)?;
    let c_dev = stream.clone_htod(c.1).map_err(e)?;
    let mut d_dev = stream
        .alloc_zeros::<f32>(d.shape.iter().product::<i64>() as usize)
        .map_err(e)?;
    let (alpha, gamma) = (1f32, 1f32);
    {
        let (pa, _ga) = a_dev.device_ptr(&stream);
        let (pc, _gc) = c_dev.device_ptr(&stream);
        let (pd, _gd) = d_dev.device_ptr_mut(&stream);
        step("elementwise_binary_execute", unsafe {
            sys::cutensorElementwiseBinaryExecute(
                h,
                plan,
                &alpha as *const f32 as *const c_void,
                pa as *const c_void,
                &gamma as *const f32 as *const c_void,
                pc as *const c_void,
                pd as *mut c_void,
                stream.cu_stream() as _,
            )
        })?;
    }
    stream.clone_dtoh(&d_dev).map_err(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2x3 row-major, and its transpose stored as 3x2 row-major.
    const X: [f32; 6] = [1., 2., 3., 4., 5., 6.];
    const X_T: [f32; 6] = [1., 4., 2., 5., 3., 6.];
    const Y: [f32; 6] = [10., 20., 30., 40., 50., 60.];
    const SUM: [f32; 6] = [11., 22., 33., 44., 55., 66.];

    fn l<'a>(shape: &'a [i64], strides: &'a [i64], modes: &'a [i32]) -> Layout<'a> {
        Layout {
            shape,
            strides,
            modes,
        }
    }

    // Requires a GPU and cuTENSOR.
    /// Same layout everywhere, D shares C's descriptor.
    #[test]
    #[ignore]
    fn same_layout_shared_d_desc() {
        let m = [0, 1];
        let a = l(&[2, 3], &[3, 1], &m);
        assert_eq!(run((&a, &X), (&a, &Y), &a, true).unwrap(), SUM);
    }

    /// Same layout everywhere, D has its own descriptor.
    #[test]
    #[ignore]
    fn same_layout_separate_d_desc() {
        let m = [0, 1];
        let a = l(&[2, 3], &[3, 1], &m);
        assert_eq!(run((&a, &X), (&a, &Y), &a, false).unwrap(), SUM);
    }

    /// A is permuted onto C's modes; D has C's modes.
    #[test]
    #[ignore]
    fn permuted_a() {
        let a = l(&[3, 2], &[2, 1], &[1, 0]);
        let c = l(&[2, 3], &[3, 1], &[0, 1]);
        assert_eq!(run((&a, &X_T), (&c, &Y), &c, false).unwrap(), SUM);
    }

    /// D's modes differ from C's modes: cuTENSOR rejects this.
    #[test]
    #[ignore]
    fn d_modes_differ_from_c() {
        let a = l(&[2, 3], &[3, 1], &[0, 1]);
        let c = l(&[3, 2], &[2, 1], &[1, 0]);
        let d = l(&[2, 3], &[3, 1], &[0, 1]);
        let r = run((&a, &X), (&c, &X_T), &d, false);
        assert!(r.unwrap_err().contains("NOT_SUPPORTED"));
    }
}
