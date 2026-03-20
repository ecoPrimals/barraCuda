// SPDX-License-Identifier: AGPL-3.0-or-later
//! Op dispatch for `TensorSession::run()`.
//!
//! Separated from `mod.rs` to keep files under the 1 000-line limit.
//! All functions here are `impl TensorSession` — Rust allows splitting
//! impl blocks across sibling modules.

use crate::device::capabilities::WORKGROUP_SIZE_1D;

use super::TensorSession;
use super::types::{
    AttentionParams, HeadReshapeParams, LayerNormParams, MatMulParams, MatMulTier, ScaleParams,
    SessionOp,
};
use bytemuck::bytes_of;

impl TensorSession {
    /// Dispatch a single recorded op into the command encoder.
    pub(super) fn dispatch_op(&self, encoder: &mut wgpu::CommandEncoder, op: &SessionOp) {
        match op {
            // ── Elementwise (pre-compiled pipelines, zero alloc) ──────────
            SessionOp::Add {
                input_a,
                input_b,
                output,
            } => {
                let size = self.elem_count(*output) as u32;
                let bg = self.auto_bind_group(
                    &self.pipelines.add_pl,
                    &[
                        &self.buffers[*input_a],
                        &self.buffers[*input_b],
                        &self.buffers[*output],
                    ],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.add_pl,
                    &bg,
                    size.div_ceil(self.workgroup_size),
                );
            }
            SessionOp::Mul {
                input_a,
                input_b,
                output,
            } => {
                let size = self.elem_count(*output) as u32;
                let bg = self.auto_bind_group(
                    &self.pipelines.mul_pl,
                    &[
                        &self.buffers[*input_a],
                        &self.buffers[*input_b],
                        &self.buffers[*output],
                    ],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.mul_pl,
                    &bg,
                    size.div_ceil(self.workgroup_size),
                );
            }
            SessionOp::Fma {
                input_a,
                input_b,
                input_c,
                output,
            } => {
                let size = self.elem_count(*output) as u32;
                let bg = self.auto_bind_group(
                    &self.pipelines.fma_pl,
                    &[
                        &self.buffers[*input_a],
                        &self.buffers[*input_b],
                        &self.buffers[*input_c],
                        &self.buffers[*output],
                    ],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.fma_pl,
                    &bg,
                    size.div_ceil(self.workgroup_size),
                );
            }
            SessionOp::Scale {
                input,
                scalar,
                output,
            } => {
                let size = self.elem_count(*output) as u32;
                let params_buf =
                    self.uniform_buf(bytes_of(&ScaleParams { scalar: *scalar }), "Scale");
                let bg = self.auto_bind_group(
                    &self.pipelines.scale_pl,
                    &[&self.buffers[*input], &params_buf, &self.buffers[*output]],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.scale_pl,
                    &bg,
                    size.div_ceil(self.workgroup_size),
                );
            }

            // ── Matrix multiply ───────────────────────────────────────────
            SessionOp::MatMul {
                input_a,
                input_b,
                output,
                m,
                k,
                n,
                tier,
            } => {
                let pipeline = match tier {
                    MatMulTier::Naive => &self.pipelines.mm_naive_pl,
                    MatMulTier::Tiled16 => &self.pipelines.mm_t16_pl,
                    MatMulTier::CpuTiled32 => &self.pipelines.mm_cpu_pl,
                    MatMulTier::GpuEvolved32 => &self.pipelines.mm_gpu_pl,
                };
                let params_buf = self.uniform_buf(
                    bytes_of(&MatMulParams {
                        m: *m,
                        k: *k,
                        n: *n,
                        _padding: 0,
                    }),
                    "MatMul",
                );
                let bg = self.auto_bind_group(
                    pipeline,
                    &[
                        &self.buffers[*input_a],
                        &self.buffers[*input_b],
                        &self.buffers[*output],
                        &params_buf,
                    ],
                );
                let (wg_x, wg_y) = tier.dispatch(*m, *n);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            // ── Activations ───────────────────────────────────────────────
            SessionOp::ReLU { input, output } => {
                let size = self.elem_count(*output) as u32;
                let bg = self.auto_bind_group(
                    &self.pipelines.relu_pl,
                    &[&self.buffers[*input], &self.buffers[*output]],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.relu_pl,
                    &bg,
                    size.div_ceil(WORKGROUP_SIZE_1D),
                );
            }
            SessionOp::Gelu { input, output } => {
                let size = self.elem_count(*output) as u32;
                let size_buf = self.uniform_u32(size, "GELU Size");
                let bg = self.auto_bind_group(
                    &self.pipelines.gelu_pl,
                    &[&self.buffers[*input], &self.buffers[*output], &size_buf],
                );
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.gelu_pl,
                    &bg,
                    size.div_ceil(WORKGROUP_SIZE_1D),
                );
            }
            SessionOp::Softmax { input, output } => {
                let size = self.elem_count(*output) as u32;
                let params_buf = self.uniform_u32(size, "Softmax Params");
                let bg = self.auto_bind_group(
                    &self.pipelines.sfmx_pl,
                    &[&self.buffers[*input], &self.buffers[*output], &params_buf],
                );
                self.dispatch_1d(&mut *encoder, &self.pipelines.sfmx_pl, &bg, 1);
            }

            // ── Layer normalisation ───────────────────────────────────────
            SessionOp::LayerNorm {
                input,
                output,
                feature_size,
            } => {
                let total = self.elem_count(*output) as u32;
                let params_buf = self.uniform_buf(
                    bytes_of(&LayerNormParams {
                        size: total,
                        feature_size: *feature_size,
                        epsilon: 1e-5,
                    }),
                    "LayerNorm",
                );
                let bg = self.auto_bind_group(
                    &self.pipelines.lnrm_pl,
                    &[&self.buffers[*input], &self.buffers[*output], &params_buf],
                );
                let rows = (total / feature_size).max(1);
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.lnrm_pl,
                    &bg,
                    rows.div_ceil(WORKGROUP_SIZE_1D),
                );
            }

            // ── Attention (3-pass SDPA) ───────────────────────────────────
            SessionOp::Attention {
                q,
                k,
                v,
                output,
                batch_size,
                num_heads,
                seq_len,
                head_dim,
            } => {
                let attn_buf = self.uniform_buf(
                    bytes_of(&AttentionParams {
                        batch_size: *batch_size,
                        num_heads: *num_heads,
                        q_seq_len: *seq_len,
                        kv_seq_len: *seq_len,
                        head_dim: *head_dim,
                        _padding: [0; 3],
                    }),
                    "Attn",
                );
                let ss_bytes = (*batch_size * *num_heads * *seq_len * *seq_len) as u64
                    * std::mem::size_of::<f32>() as u64;
                let scores_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Attn Scores"),
                    size: ss_bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                let weights_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Attn Weights"),
                    size: ss_bytes.max(4),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let bg1 = self.auto_bind_group(
                    &self.pipelines.sdpa_scores_pl,
                    &[&self.buffers[*q], &self.buffers[*k], &scores_buf, &attn_buf],
                );
                let total1 = *batch_size * *num_heads * *seq_len * *seq_len;
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.sdpa_scores_pl,
                    &bg1,
                    total1.div_ceil(WORKGROUP_SIZE_1D),
                );

                let bg2 = self.auto_bind_group(
                    &self.pipelines.attn_softmax_pl,
                    &[&scores_buf, &weights_buf, &attn_buf],
                );
                let total2 = *batch_size * *num_heads * *seq_len;
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.attn_softmax_pl,
                    &bg2,
                    total2.div_ceil(WORKGROUP_SIZE_1D),
                );

                let bg3 = self.auto_bind_group(
                    &self.pipelines.attn_apply_pl,
                    &[
                        &weights_buf,
                        &self.buffers[*v],
                        &self.buffers[*output],
                        &attn_buf,
                    ],
                );
                {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&self.pipelines.attn_apply_pl);
                    pass.set_bind_group(0, Some(&bg3), &[]);
                    pass.dispatch_workgroups(
                        head_dim.div_ceil(16),
                        seq_len.div_ceil(16),
                        *batch_size * *num_heads,
                    );
                }
            }

            // ── Head reshape ──────────────────────────────────────────────
            SessionOp::HeadSplit {
                input,
                output,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            } => {
                let p = self.uniform_buf(
                    bytes_of(&HeadReshapeParams {
                        batch_size: *batch_size,
                        seq_len: *seq_len,
                        num_heads: *num_heads,
                        head_dim: *head_dim,
                    }),
                    "HeadSplit",
                );
                let bg = self.auto_bind_group(
                    &self.pipelines.head_split_pl,
                    &[&self.buffers[*input], &self.buffers[*output], &p],
                );
                let total = *batch_size * *num_heads * *seq_len * *head_dim;
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.head_split_pl,
                    &bg,
                    total.div_ceil(WORKGROUP_SIZE_1D),
                );
            }
            SessionOp::HeadConcat {
                input,
                output,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            } => {
                let p = self.uniform_buf(
                    bytes_of(&HeadReshapeParams {
                        batch_size: *batch_size,
                        seq_len: *seq_len,
                        num_heads: *num_heads,
                        head_dim: *head_dim,
                    }),
                    "HeadConcat",
                );
                let bg = self.auto_bind_group(
                    &self.pipelines.head_concat_pl,
                    &[&self.buffers[*input], &self.buffers[*output], &p],
                );
                let total = *batch_size * *num_heads * *seq_len * *head_dim;
                self.dispatch_1d(
                    encoder,
                    &self.pipelines.head_concat_pl,
                    &bg,
                    total.div_ceil(WORKGROUP_SIZE_1D),
                );
            }
        }
    }

    // ── Dispatch-only helpers ─────────────────────────────────────────────────

    pub(super) fn elem_count(&self, id: usize) -> usize {
        self.shapes[id].iter().product()
    }

    /// Bind group derived from the pipeline's auto-reflected layout at group 0.
    pub(super) fn auto_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            })
    }

    /// Dispatch a pipeline with a single 1-D workgroup count.
    pub(super) fn dispatch_1d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let wg = workgroups.clamp(1, 65535);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bg), &[]);
        pass.dispatch_workgroups(wg, 1, 1);
    }

    /// Create a small uniform buffer containing raw bytes.
    pub(super) fn uniform_buf(&self, bytes: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Create a uniform buffer containing a single `u32`.
    pub(super) fn uniform_u32(&self, value: u32, label: &str) -> wgpu::Buffer {
        self.uniform_buf(bytemuck::bytes_of(&value), label)
    }
}
