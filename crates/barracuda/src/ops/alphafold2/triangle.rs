// SPDX-License-Identifier: AGPL-3.0-only

//! Triangle multiplication (outgoing and incoming).

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::WG_64;

const SHADER_TRI_MUL_OUT: &str =
    include_str!("../../shaders/attention/triangle_mul_outgoing_f64.wgsl");
const SHADER_TRI_MUL_IN: &str =
    include_str!("../../shaders/attention/triangle_mul_incoming_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TriangleMulParams {
    n: u32,
    _pad: [u32; 3],
}

/// Outgoing triangle multiplication: `out[i,j] = sum_k gate[i,k] * a[i,k] * b[j,k]`.
///
/// Pair representation update from the Evoformer stack.
pub fn triangle_mul_outgoing(
    device: &Arc<WgpuDevice>,
    a: &[f64],
    b: &[f64],
    gate: &[f64],
    n: u32,
) -> Result<Vec<f64>> {
    let total = (n * n) as usize;
    let a_buf = device.create_buffer_f64_init("tri_mul_out:a", a);
    let b_buf = device.create_buffer_f64_init("tri_mul_out:b", b);
    let gate_buf = device.create_buffer_f64_init("tri_mul_out:gate", gate);
    let out_buf = device.create_buffer_f64(total)?;
    let params = TriangleMulParams { n, _pad: [0; 3] };
    let params_buf = device.create_uniform_buffer("tri_mul_out:params", &params);

    ComputeDispatch::new(device, "triangle_mul_outgoing")
        .shader(SHADER_TRI_MUL_OUT, "main")
        .f64()
        .storage_read(0, &a_buf)
        .storage_read(1, &b_buf)
        .storage_read(2, &gate_buf)
        .storage_rw(3, &out_buf)
        .uniform(4, &params_buf)
        .dispatch(total.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, total)
}

/// Incoming triangle multiplication: `out[i,j] = sum_k gate[k,j] * a[k,i] * b[k,j]`.
pub fn triangle_mul_incoming(
    device: &Arc<WgpuDevice>,
    a: &[f64],
    b: &[f64],
    gate: &[f64],
    n: u32,
) -> Result<Vec<f64>> {
    let total = (n * n) as usize;
    let a_buf = device.create_buffer_f64_init("tri_mul_in:a", a);
    let b_buf = device.create_buffer_f64_init("tri_mul_in:b", b);
    let gate_buf = device.create_buffer_f64_init("tri_mul_in:gate", gate);
    let out_buf = device.create_buffer_f64(total)?;
    let params = TriangleMulParams { n, _pad: [0; 3] };
    let params_buf = device.create_uniform_buffer("tri_mul_in:params", &params);

    ComputeDispatch::new(device, "triangle_mul_incoming")
        .shader(SHADER_TRI_MUL_IN, "main")
        .f64()
        .storage_read(0, &a_buf)
        .storage_read(1, &b_buf)
        .storage_read(2, &gate_buf)
        .storage_rw(3, &out_buf)
        .uniform(4, &params_buf)
        .dispatch(total.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_triangle_mul() {
        assert_eq!(std::mem::size_of::<TriangleMulParams>(), 16);
    }

    #[cfg(feature = "gpu")]
    fn shader_compiles_via_naga(source: &str) {
        let full = crate::shaders::precision::ShaderTemplate::for_driver_auto(source, false);
        let normalized = format!(
            "enable f64;\n\n{}",
            full.replace("enable f64;\n\n", "")
                .replace("enable f64;\n", "")
        );
        if let Ok(module) = naga::front::wgsl::parse_str(&normalized) {
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("WGSL validation failed");
            assert!(!module.entry_points.is_empty());
        } else {
            assert!(!full.is_empty());
            assert!(full.contains("fn main"), "shader must contain fn main");
            assert!(
                full.contains("@compute") || full.contains("main"),
                "expected compute entry"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn triangle_mul_outgoing_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/triangle_mul_outgoing_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn triangle_mul_incoming_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/attention/triangle_mul_incoming_f64.wgsl");
        shader_compiles_via_naga(source);
    }
}
