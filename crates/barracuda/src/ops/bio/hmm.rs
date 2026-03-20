// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batch HMM algorithms (f64, GPU) — forward, backward, Viterbi.
//!
//! One thread per observation sequence. Sequential over T time steps within
//! each sequence, parallel across B sequences. All computation in log-domain
//! with `log_sum_exp2` for numerical stability.
//!
//! ## Absorbed from
//!
//! wetSpring handoff v6, `hmm_forward_f64.wgsl` — 13/13 GPU checks PASS.
//! neuralSpring S69, `hmm_backward_log_f64.wgsl` + `hmm_viterbi_f64.wgsl`.
//! Polyfill required for Ada Lovelace (uses f64 exp/log in `log_sum_exp2`).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "per-timestep shader kept for host-side multi-dispatch HMM loop"
    )
)]
const SHADER_PER_STEP: &str = include_str!("../../shaders/bio/hmm_forward_f64.wgsl");
const SHADER: &str = include_str!("../../shaders/bio/hmm_batch_forward_f64.wgsl");
const SHADER_BACKWARD: &str = include_str!("../../shaders/bio/hmm_backward_log_f64.wgsl");
const SHADER_VITERBI: &str = include_str!("../../shaders/bio/hmm_viterbi_f64.wgsl");

/// Log-domain f32 HMM forward shader (neuralSpring metalForge provenance).
///
/// Uses max-subtract trick for numerical stability. One thread per destination
/// state; lighter-weight than the f64 variant above, suitable for real-time
/// inference where f32 precision suffices.
pub const WGSL_HMM_FORWARD_LOG_F32: &str = include_str!("../../shaders/ml/hmm_forward_log.wgsl");

/// f64 version of the log-domain HMM forward pass for universal math library.
/// Wired and ready; no separate log-domain pipeline in this module — `HmmBatchForwardF64`
/// uses the main `hmm_forward_f64.wgsl` shader via `compile_shader_f64`.
pub const WGSL_HMM_FORWARD_LOG_F64: &str =
    include_str!("../../shaders/ml/hmm_forward_log_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmParams {
    n_states: u32,
    n_symbols: u32,
    n_steps: u32,
    n_seqs: u32,
}

/// GPU-resident buffers for an HMM forward pass dispatch.
///
/// Groups dimensions and buffer references to avoid 11-argument function calls.
pub struct HmmForwardArgs<'a> {
    /// Number of hidden states.
    pub n_states: u32,
    /// Number of emission symbols.
    pub n_symbols: u32,
    /// Number of time steps per sequence.
    pub n_steps: u32,
    /// Number of sequences in the batch.
    pub n_seqs: u32,
    /// Log transition matrix (N×N).
    pub log_trans: &'a wgpu::Buffer,
    /// Log emission matrix (N×M).
    pub log_emit: &'a wgpu::Buffer,
    /// Log initial state probabilities (N).
    pub log_pi: &'a wgpu::Buffer,
    /// Observation sequences (B×T).
    pub observations: &'a wgpu::Buffer,
    /// Output: log forward variables (B×T×N).
    pub log_alpha_out: &'a wgpu::Buffer,
    /// Output: log likelihood per sequence (B).
    pub log_lik_out: &'a wgpu::Buffer,
}

/// Batch HMM forward pass on GPU (log-domain, f64).
pub struct HmmBatchForwardF64 {
    device: Arc<WgpuDevice>,
}

impl HmmBatchForwardF64 {
    /// Create batch HMM forward pipeline for the given device.
    /// # Errors
    /// Returns [`Err`] if device initialization or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch the forward pass on GPU-resident buffers.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, the device is lost, or compute
    /// submission fails.
    pub fn dispatch(&self, args: &HmmForwardArgs<'_>) -> Result<()> {
        let params = HmmParams {
            n_states: args.n_states,
            n_symbols: args.n_symbols,
            n_steps: args.n_steps,
            n_seqs: args.n_seqs,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("HmmForward:params", &params);

        let wg_count = args.n_seqs.div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "hmm_forward")
            .shader(SHADER, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, args.log_trans)
            .storage_read(2, args.log_emit)
            .storage_read(3, args.log_pi)
            .storage_read(4, args.observations)
            .storage_rw(5, args.log_alpha_out)
            .storage_rw(6, args.log_lik_out)
            .dispatch(wg_count, 1, 1)
            .submit()?;
        Ok(())
    }
}

// ── HMM Backward (log-domain, ComputeDispatch) ─────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmBackwardParams {
    t_steps: u32,
    n_states: u32,
}

/// Batch HMM backward pass (β) on GPU via `ComputeDispatch`.
///
/// Computes log-domain backward variables for posterior decoding.
/// Single workgroup — sequential over time, parallel across states.
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
/// readback fails.
pub fn hmm_backward(
    device: &Arc<WgpuDevice>,
    log_trans: &[f64],
    log_emit: &[f64],
    t_steps: u32,
    n_states: u32,
) -> Result<Vec<f64>> {
    let out_len = (t_steps * n_states) as usize;
    let trans_buf = device.create_buffer_f64_init("hmm_bwd:trans", log_trans);
    let emit_buf = device.create_buffer_f64_init("hmm_bwd:emit", log_emit);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = HmmBackwardParams { t_steps, n_states };
    let params_buf = device.create_uniform_buffer("hmm_bwd:params", &params);

    ComputeDispatch::new(device, "hmm_backward")
        .shader(SHADER_BACKWARD, "main")
        .f64()
        .storage_read(0, &trans_buf)
        .storage_read(1, &emit_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(1, 1, 1)
        .submit()?;

    device.read_f64_buffer(&out_buf, out_len)
}

// ── HMM Viterbi Decoding (ComputeDispatch) ──────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmViterbiParams {
    t_steps: u32,
    n_states: u32,
}

/// Viterbi decoding result.
#[derive(Debug, Clone)]
pub struct ViterbiResult {
    /// Most likely state sequence.
    pub path: Vec<u32>,
    /// Viterbi delta values (max log-prob at each step).
    pub delta: Vec<f64>,
    /// Backpointer (psi) for traceback.
    pub psi: Vec<u32>,
}

/// Batch HMM Viterbi decoding on GPU via `ComputeDispatch`.
///
/// Finds the maximum-likelihood state sequence. Single workgroup dispatch
/// (sequential over time steps).
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation fails, the device is lost, or buffer
/// readback fails.
pub fn hmm_viterbi(
    device: &Arc<WgpuDevice>,
    log_trans: &[f64],
    log_emit: &[f64],
    log_init: &[f64],
    t_steps: u32,
    n_states: u32,
) -> Result<ViterbiResult> {
    let path_len = t_steps as usize;
    let delta_len = (t_steps * n_states) as usize;
    let psi_len = delta_len;

    let trans_buf = device.create_buffer_f64_init("viterbi:trans", log_trans);
    let emit_buf = device.create_buffer_f64_init("viterbi:emit", log_emit);
    let init_buf = device.create_buffer_f64_init("viterbi:init", log_init);
    let path_buf = device.create_buffer_u32(path_len)?;
    let delta_buf = device.create_buffer_f64(delta_len)?;
    let psi_buf = device.create_buffer_u32(psi_len)?;
    let params = HmmViterbiParams { t_steps, n_states };
    let params_buf = device.create_uniform_buffer("viterbi:params", &params);

    ComputeDispatch::new(device, "hmm_viterbi")
        .shader(SHADER_VITERBI, "main")
        .f64()
        .storage_read(0, &trans_buf)
        .storage_read(1, &emit_buf)
        .storage_read(2, &init_buf)
        .storage_rw(3, &path_buf)
        .storage_rw(4, &delta_buf)
        .storage_rw(5, &psi_buf)
        .uniform(6, &params_buf)
        .dispatch(1, 1, 1)
        .submit()?;

    let path = device.read_buffer_u32(&path_buf, path_len)?;
    let delta = device.read_f64_buffer(&delta_buf, delta_len)?;
    let psi = device.read_buffer_u32(&psi_buf, psi_len)?;

    Ok(ViterbiResult { path, delta, psi })
}

// ── HMM Forward Log-Domain (metalForge / neuralSpring) ──────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HmmForwardLogParams {
    n_states: u32,
}

/// Log-domain HMM forward pass (f32) — single step.
///
/// Uses max-subtract trick. One thread per destination state. Suitable for
/// real-time inference where f32 precision suffices.
pub struct HmmForwardLogF32 {
    device: Arc<WgpuDevice>,
}

impl HmmForwardLogF32 {
    /// Create log-domain HMM forward (f32) pipeline.
    /// # Errors
    /// Returns [`Err`] if device initialization or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch one forward step: `alpha_curr` = `fwd_step(alpha_prev`, `log_trans`, `log_emit`).
    /// # Errors
    /// Returns [`Err`] if the device is lost or compute submission fails.
    pub fn dispatch(
        &self,
        n_states: u32,
        alpha_prev: &wgpu::Buffer,
        log_trans: &wgpu::Buffer,
        log_emit: &wgpu::Buffer,
        alpha_curr: &wgpu::Buffer,
    ) -> Result<()> {
        let params = HmmForwardLogParams { n_states };
        let params_buf = self
            .device
            .create_uniform_buffer("HmmForwardLogF32:params", &params);

        let wg_count = n_states.div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "hmm_forward_log_f32")
            .shader(WGSL_HMM_FORWARD_LOG_F32, "hmm_forward_log")
            .storage_read(0, alpha_prev)
            .storage_read(1, log_trans)
            .storage_read(2, log_emit)
            .storage_rw(3, alpha_curr)
            .uniform(4, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;
        Ok(())
    }
}

/// Log-domain HMM forward pass (f64) — single step.
///
/// Uses max-subtract trick. One thread per destination state.
pub struct HmmForwardLogF64 {
    device: Arc<WgpuDevice>,
}

impl HmmForwardLogF64 {
    /// Create log-domain HMM forward (f64) pipeline.
    /// # Errors
    /// Returns [`Err`] if device initialization or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch one forward step: `alpha_curr` = `fwd_step(alpha_prev`, `log_trans`, `log_emit`).
    /// # Errors
    /// Returns [`Err`] if the device is lost or compute submission fails.
    pub fn dispatch(
        &self,
        n_states: u32,
        alpha_prev: &wgpu::Buffer,
        log_trans: &wgpu::Buffer,
        log_emit: &wgpu::Buffer,
        alpha_curr: &wgpu::Buffer,
    ) -> Result<()> {
        let params = HmmForwardLogParams { n_states };
        let params_buf = self
            .device
            .create_uniform_buffer("HmmForwardLogF64:params", &params);

        let wg_count = n_states.div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "hmm_forward_log_f64")
            .shader(WGSL_HMM_FORWARD_LOG_F64, "hmm_forward_log")
            .f64()
            .storage_read(0, alpha_prev)
            .storage_read(1, log_trans)
            .storage_read(2, log_emit)
            .storage_rw(3, alpha_curr)
            .uniform(4, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_source_valid() {
        assert!(
            SHADER.contains("logsumexp")
                || SHADER.contains("log_sum_exp")
                || SHADER.contains("sum_exp"),
            "batch forward shader must implement logsumexp"
        );
        assert!(SHADER.contains("Params"));
        assert!(SHADER.contains("log_lik_out"));
        assert!(SHADER.contains("log_pi"));
    }

    #[test]
    fn per_step_shader_source_valid() {
        assert!(SHADER_PER_STEP.contains("log_alpha_prev"));
        assert!(SHADER_PER_STEP.contains("log_alpha_next"));
    }

    #[test]
    fn backward_shader_source_valid() {
        assert!(SHADER_BACKWARD.contains("HmmBackwardParams"));
    }

    #[test]
    fn viterbi_shader_source_valid() {
        assert!(SHADER_VITERBI.contains("HmmViterbiParams"));
    }

    #[test]
    fn test_backward_shader_contains_entrypoint() {
        assert!(SHADER_BACKWARD.contains("@compute"));
        assert!(SHADER_BACKWARD.contains("fn main"));
    }

    #[test]
    fn test_viterbi_shader_contains_entrypoint() {
        assert!(SHADER_VITERBI.contains("@compute"));
        assert!(SHADER_VITERBI.contains("fn main"));
    }

    #[test]
    fn params_layout_backward() {
        assert_eq!(std::mem::size_of::<HmmBackwardParams>(), 8);
    }

    #[test]
    fn params_layout_viterbi() {
        assert_eq!(std::mem::size_of::<HmmViterbiParams>(), 8);
    }

    #[test]
    fn log_f32_shader_source_valid() {
        assert!(WGSL_HMM_FORWARD_LOG_F32.contains("hmm_forward_log"));
        assert!(WGSL_HMM_FORWARD_LOG_F32.contains("HmmParams"));
        assert!(WGSL_HMM_FORWARD_LOG_F32.contains("alpha_prev"));
    }

    #[test]
    fn log_f64_shader_source_valid() {
        assert!(WGSL_HMM_FORWARD_LOG_F64.contains("hmm_forward_log"));
        assert!(WGSL_HMM_FORWARD_LOG_F64.contains("HmmParams"));
        assert!(WGSL_HMM_FORWARD_LOG_F64.contains("alpha_prev"));
    }

    #[test]
    fn params_layout_forward_log() {
        assert_eq!(std::mem::size_of::<HmmForwardLogParams>(), 4);
    }
}
