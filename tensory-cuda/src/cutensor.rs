//! Thin RAII layer over cudarc's raw cuTENSOR bindings.

use std::{ffi::c_void, mem::MaybeUninit};

use cudarc::{
    cutensor::{
        result::{self, CutensorError},
        sys,
    },
    driver::{CudaSlice, DevicePtr, DevicePtrMut},
};

use crate::{
    CudaDenseRepr, CudaErr, CudaRuntime, CudaScalar, contiguous_strides, dense_strides_like,
};

// Lower than the actual allocation alignment, hence always valid.
const ALIGNMENT: u32 = 16;

pub(crate) struct Handle(sys::cutensorHandle_t);

impl Handle {
    pub(crate) fn new() -> Result<Self, CutensorError> {
        Ok(Self(result::create_handle()?))
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        let _ = unsafe { result::destroy_handle(self.0) };
    }
}

struct Owned<T: Copy>(T, unsafe fn(T) -> Result<(), CutensorError>);

impl<T: Copy> Drop for Owned<T> {
    fn drop(&mut self) {
        let _ = unsafe { (self.1)(self.0) };
    }
}

/// A tensor operand: device data, its layout and its mode labels.
pub(crate) struct Operand<'a, E> {
    pub data: &'a CudaSlice<E>,
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub modes: Vec<i32>,
}

fn check(status: sys::cutensorStatus_t) -> Result<(), CutensorError> {
    match status {
        sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS => Ok(()),
        s => Err(CutensorError(s)),
    }
}

fn tensor_desc<E: CudaScalar>(
    handle: &Handle,
    shape: &[usize],
    strides: &[usize],
) -> Result<Owned<sys::cutensorTensorDescriptor_t>, CutensorError> {
    let extent: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
    let stride: Vec<i64> = strides.iter().map(|&x| x as i64).collect();
    let desc = unsafe {
        result::create_tensor_descriptor(
            handle.0,
            shape.len() as u32,
            extent.as_ptr(),
            stride.as_ptr(),
            E::DATA_TYPE,
            ALIGNMENT,
        )
    }?;
    Ok(Owned(desc, result::destroy_tensor_descriptor))
}

fn make_plan(
    handle: &Handle,
    op: sys::cutensorOperationDescriptor_t,
    needs_workspace: bool,
) -> Result<(Owned<sys::cutensorPlan_t>, u64), CutensorError> {
    unsafe {
        let pref = Owned(
            result::create_plan_preference(
                handle.0,
                sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
                sys::cutensorJitMode_t::CUTENSOR_JIT_MODE_NONE,
            )?,
            result::destroy_plan_preference,
        );
        // Estimating a workspace is not supported for elementwise operations.
        let workspace = if needs_workspace {
            result::estimate_workspace_size(
                handle.0,
                op,
                pref.0,
                sys::cutensorWorksizePreference_t::CUTENSOR_WORKSPACE_DEFAULT,
            )?
        } else {
            0
        };
        let plan = result::create_plan(handle.0, op, pref.0, workspace)?;
        Ok((Owned(plan, result::destroy_plan), workspace))
    }
}

impl CudaRuntime {
    /// `out[out_modes] = sum_{contracted modes} a[a.modes] * b[b.modes]`.
    pub(crate) fn contract<E: CudaScalar>(
        &self,
        a: Operand<'_, E>,
        b: Operand<'_, E>,
        out_shape: Vec<usize>,
        out_modes: &[i32],
    ) -> Result<CudaDenseRepr<E>, CudaErr> {
        let out_strides = contiguous_strides(&out_shape);
        let mut out = self.stream.alloc_zeros::<E>(out_shape.iter().product())?;
        // An empty operand or output makes the result all zeros.
        if out.is_empty() || a.data.is_empty() || b.data.is_empty() {
            return Ok(CudaDenseRepr::from_raw_layout(out, out_shape, out_strides));
        }
        self.stream.context().bind_to_thread()?;

        let h = &*self.handle;
        let desc_a = tensor_desc::<E>(h, a.shape, a.strides)?;
        let desc_b = tensor_desc::<E>(h, b.shape, b.strides)?;
        let desc_out = tensor_desc::<E>(h, &out_shape, &out_strides)?;
        let identity = sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY;
        let op = Owned(
            unsafe {
                result::create_contraction(
                    h.0,
                    desc_a.0,
                    a.modes.as_ptr(),
                    identity,
                    desc_b.0,
                    b.modes.as_ptr(),
                    identity,
                    desc_out.0,
                    out_modes.as_ptr(),
                    identity,
                    desc_out.0,
                    out_modes.as_ptr(),
                    E::compute_desc(),
                )
            }?,
            result::destroy_operation_descriptor,
        );
        let (plan, workspace_size) = make_plan(h, op.0, true)?;
        let mut workspace = self
            .stream
            .alloc_zeros::<u8>(workspace_size.max(1) as usize)?;

        let (alpha, beta) = (E::ONE, E::ZERO);
        {
            let s = &self.stream;
            let (pa, _ga) = a.data.device_ptr(s);
            let (pb, _gb) = b.data.device_ptr(s);
            let (po, _go) = out.device_ptr_mut(s);
            let (pw, _gw) = workspace.device_ptr_mut(s);
            unsafe {
                result::contract(
                    h.0,
                    plan.0,
                    &alpha as *const E as *const c_void,
                    pa as *const c_void,
                    pb as *const c_void,
                    &beta as *const E as *const c_void,
                    po as *const c_void,
                    po as *mut c_void,
                    pw as *mut c_void,
                    workspace_size,
                    s.cu_stream() as _,
                )
            }?;
        }
        Ok(CudaDenseRepr::from_raw_layout(out, out_shape, out_strides))
    }

    /// `out = alpha * a + gamma * c` where `a` is permuted onto `c`'s modes.
    /// cuTENSOR requires D to have C's modes, so `out` has `c`'s axis order and memory order.
    pub(crate) fn add_scaled<E: CudaScalar>(
        &self,
        a: Operand<'_, E>,
        alpha: E,
        c: Operand<'_, E>,
        gamma: E,
    ) -> Result<CudaDenseRepr<E>, CudaErr> {
        let out_shape = c.shape.to_vec();
        let out_modes = c.modes.clone();
        let out_strides = dense_strides_like(&out_shape, c.strides);
        let mut out = self.stream.alloc_zeros::<E>(out_shape.iter().product())?;
        if out.is_empty() {
            return Ok(CudaDenseRepr::from_raw_layout(out, out_shape, out_strides));
        }
        self.stream.context().bind_to_thread()?;

        let h = &*self.handle;
        let desc_a = tensor_desc::<E>(h, a.shape, a.strides)?;
        let desc_c = tensor_desc::<E>(h, c.shape, c.strides)?;
        let desc_out = tensor_desc::<E>(h, &out_shape, &out_strides)?;
        let identity = sys::cutensorOperator_t::CUTENSOR_OP_IDENTITY;
        let mut raw_op = MaybeUninit::uninit();
        check(unsafe {
            sys::cutensorCreateElementwiseBinary(
                h.0,
                raw_op.as_mut_ptr(),
                desc_a.0,
                a.modes.as_ptr(),
                identity,
                desc_c.0,
                c.modes.as_ptr(),
                identity,
                desc_out.0,
                out_modes.as_ptr(),
                sys::cutensorOperator_t::CUTENSOR_OP_ADD,
                E::compute_desc(),
            )
        })?;
        let op = Owned(
            unsafe { raw_op.assume_init() },
            result::destroy_operation_descriptor,
        );
        let (plan, _) = make_plan(h, op.0, false)?;

        {
            let s = &self.stream;
            let (pa, _ga) = a.data.device_ptr(s);
            let (pc, _gc) = c.data.device_ptr(s);
            let (po, _go) = out.device_ptr_mut(s);
            check(unsafe {
                sys::cutensorElementwiseBinaryExecute(
                    h.0,
                    plan.0,
                    &alpha as *const E as *const c_void,
                    pa as *const c_void,
                    &gamma as *const E as *const c_void,
                    pc as *const c_void,
                    po as *mut c_void,
                    s.cu_stream() as _,
                )
            })?;
        }
        Ok(CudaDenseRepr::from_raw_layout(out, out_shape, out_strides))
    }
}
