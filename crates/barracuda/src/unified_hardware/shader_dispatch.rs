// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader-first dispatch abstraction — run WGSL on GPU or CPU through one trait.
//!
//! `ShaderDispatch` decouples *what math to run* (WGSL source) from *where to run it*
//! (GPU via wgpu, CPU via naga-exec). This enables the shader-as-truth architecture
//! where WGSL is the single source of math, executable on any substrate.
//!
//! # Batch dispatch helpers
//!
//! Common dispatch patterns are provided as convenience functions:
//! - [`shader_batch_unary_f64`]: 2-binding layout (input, output) using `arrayLength`
//! - [`shader_batch_unary_f64_with_size`]: 3-binding layout with `Metadata { size }` uniform
//!
//! # Migrated modules
//!
//! The following modules dispatch through WGSL when `cpu-shader` is enabled:
//! - `activations`: relu, sigmoid, gelu, swish batch functions
//! - `special::erf`: `erf_batch`, `erfc_batch`
//! - `special::bessel`: `bessel_j0`/`j1`/`i0`/`k0` batch functions
//! - `stats::jackknife`: leave-one-out mean computation
//! - `numerical::gradient`: `gradient_1d` stencil
//! - `health::biosignal`: `convolve_1d`

#[cfg(feature = "cpu-shader")]
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::Result;

/// Binding slot for shader dispatch: `(group, binding_index, data)`.
///
/// Data is mutably borrowed — the dispatcher reads inputs and writes outputs
/// in-place, matching GPU buffer semantics.
pub struct ShaderBinding<'a> {
    /// Bind group index.
    pub group: u32,
    /// Binding index within the group.
    pub binding: u32,
    /// Raw byte data — read for inputs, written for outputs.
    pub data: &'a mut [u8],
    /// Whether this binding is read-only.
    pub read_only: bool,
}

/// Unified WGSL dispatch — GPU and CPU implement the same interface.
///
/// Callers provide WGSL source, entry point name, buffer bindings, and
/// workgroup counts. The implementation handles compilation and execution
/// on its native substrate (wgpu for GPU, naga-exec for CPU).
pub trait ShaderDispatch: Send + Sync {
    /// Execute a WGSL compute shader.
    ///
    /// # Arguments
    ///
    /// * `wgsl` — WGSL source code
    /// * `entry` — Compute entry point name (typically `"main"`)
    /// * `bindings` — Mutable buffer bindings keyed by `(group, binding)`
    /// * `workgroups` — `(x, y, z)` workgroup dispatch counts
    ///
    /// # Errors
    ///
    /// Returns errors for parse/validation failures, missing bindings,
    /// unsupported IR nodes, or backend-specific dispatch failures.
    fn dispatch_wgsl(
        &self,
        wgsl: &str,
        entry: &str,
        bindings: &mut [ShaderBinding<'_>],
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Human-readable name for diagnostics.
    fn substrate_name(&self) -> &'static str;
}

/// CPU shader dispatch via naga-exec — interprets WGSL without a GPU.
///
/// Enabled by the `cpu-shader` feature. Parses and validates WGSL once per
/// call (stateless), then interprets the compute entry point across all
/// workgroup invocations.
#[cfg(feature = "cpu-shader")]
pub struct CpuShaderDispatch;

#[cfg(feature = "cpu-shader")]
impl CpuShaderDispatch {
    /// Create a new CPU shader dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "cpu-shader")]
impl Default for CpuShaderDispatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cpu-shader")]
impl ShaderDispatch for CpuShaderDispatch {
    fn dispatch_wgsl(
        &self,
        wgsl: &str,
        entry: &str,
        bindings: &mut [ShaderBinding<'_>],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        use barracuda_naga_exec::{NagaExecutor, SimBuffer, SimBufferUsage};
        use std::collections::BTreeMap;

        let executor = NagaExecutor::new(wgsl, entry)
            .map_err(|e| crate::error::BarracudaError::device(format!("naga-exec: {e}")))?;

        let mut sim_bindings: BTreeMap<(u32, u32), SimBuffer> = BTreeMap::new();
        for b in bindings.iter() {
            let usage = if b.read_only {
                SimBufferUsage::StorageReadOnly
            } else {
                SimBufferUsage::Storage
            };
            sim_bindings.insert(
                (b.group, b.binding),
                SimBuffer {
                    data: b.data.to_vec(),
                    usage,
                },
            );
        }

        executor
            .dispatch(workgroups, &mut sim_bindings)
            .map_err(|e| {
                crate::error::BarracudaError::device(format!("naga-exec dispatch: {e}"))
            })?;

        for b in bindings.iter_mut() {
            if !b.read_only {
                if let Some(sim) = sim_bindings.get(&(b.group, b.binding)) {
                    let copy_len = b.data.len().min(sim.data.len());
                    b.data[..copy_len].copy_from_slice(&sim.data[..copy_len]);
                }
            }
        }

        Ok(())
    }

    fn substrate_name(&self) -> &'static str {
        "CPU (naga-exec)"
    }
}

// ── Batch dispatch helpers ──────────────────────────────────────────────────
//
// Convenience functions for the common patterns:
//   - unary f64: one input array → one output array (2 bindings)
//   - unary f64 with size: same but with a `Metadata { size }` uniform (3 bindings)

/// Dispatch a unary f64 shader with the 2-binding layout:
///   `@binding(0) read input`, `@binding(1) read_write output`.
///
/// Falls back to `Err` on parse/validation/execution failure so callers
/// can use native Rust as a fallback.
///
/// # Errors
///
/// Returns an error if WGSL parsing, validation, or dispatch fails.
#[cfg(feature = "cpu-shader")]
pub fn shader_batch_unary_f64(wgsl: &str, entry: &str, input: &[f64]) -> Result<Vec<f64>> {
    let dispatcher = CpuShaderDispatch::new();
    let mut in_buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out_buf = vec![0u8; in_buf.len()];
    let n = input.len() as u32;
    let workgroups = (n.div_ceil(WORKGROUP_SIZE_1D), 1, 1);

    let mut bindings = vec![
        ShaderBinding {
            group: 0,
            binding: 0,
            data: &mut in_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 1,
            data: &mut out_buf,
            read_only: false,
        },
    ];

    dispatcher.dispatch_wgsl(wgsl, entry, &mut bindings, workgroups)?;

    Ok(out_buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Dispatch a unary f64 shader with 3-binding layout.
///
/// Uses the pattern common to special-function shaders:
/// `@binding(0) read input`, `@binding(1) read_write output`,
/// `@binding(2) uniform Metadata { size: u32, _pad: u32×3 }`.
///
/// # Errors
///
/// Returns an error if WGSL parsing, validation, or dispatch fails.
#[cfg(feature = "cpu-shader")]
pub fn shader_batch_unary_f64_with_size(
    wgsl: &str,
    entry: &str,
    input: &[f64],
) -> Result<Vec<f64>> {
    let dispatcher = CpuShaderDispatch::new();
    let mut in_buf: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out_buf = vec![0u8; in_buf.len()];
    let n = input.len() as u32;
    let workgroups = (n.div_ceil(WORKGROUP_SIZE_1D), 1, 1);

    let mut meta_buf = vec![0u8; 16]; // Metadata { size: u32, _pad0, _pad1, _pad2 }
    meta_buf[..4].copy_from_slice(&n.to_le_bytes());

    let mut bindings = vec![
        ShaderBinding {
            group: 0,
            binding: 0,
            data: &mut in_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 1,
            data: &mut out_buf,
            read_only: false,
        },
        ShaderBinding {
            group: 0,
            binding: 2,
            data: &mut meta_buf,
            read_only: true,
        },
    ];

    dispatcher.dispatch_wgsl(wgsl, entry, &mut bindings, workgroups)?;

    Ok(out_buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

#[cfg(all(test, feature = "cpu-shader"))]
mod tests {
    use super::*;

    #[test]
    fn cpu_shader_dispatch_elementwise_add() {
        let dispatcher = CpuShaderDispatch::new();
        let wgsl = r"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = a[gid.x] + b[gid.x];
}
";
        let a_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = [10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out_data = vec![0u8; 16];

        let mut a_buf = a_data;
        let mut b_buf = b_data;
        let mut bindings = vec![
            ShaderBinding {
                group: 0,
                binding: 0,
                data: &mut a_buf,
                read_only: true,
            },
            ShaderBinding {
                group: 0,
                binding: 1,
                data: &mut b_buf,
                read_only: true,
            },
            ShaderBinding {
                group: 0,
                binding: 2,
                data: &mut out_data,
                read_only: false,
            },
        ];

        dispatcher
            .dispatch_wgsl(wgsl, "main", &mut bindings, (4, 1, 1))
            .unwrap();

        let result: Vec<f32> = out_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn cpu_shader_dispatch_substrate_name() {
        let d = CpuShaderDispatch::new();
        assert_eq!(d.substrate_name(), "CPU (naga-exec)");
    }
}
