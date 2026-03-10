// SPDX-License-Identifier: AGPL-3.0-only

//! Batch Inverse Participation Ratio (IPR) — GPU kernel.
//!
//! IPR measures eigenvector localization:
//!   IPR = Σ |`ψ_i|⁴`
//!
//! - Extended states: IPR ~ 1/dim
//! - Localized states: IPR >> 1/dim
//!
//! Each thread processes one eigenvector from a contiguous batch.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;

/// WGSL shader source for batch Inverse Participation Ratio (f64 downcast to f32 when needed).
pub static WGSL_BATCH_IPR: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| include_str!("../shaders/spectral/batch_ipr_f64.wgsl").to_string());

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct IprParams {
    dim: u32,
    n_vectors: u32,
}

/// GPU compute pipeline for batch Inverse Participation Ratio.
pub struct BatchIprGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl BatchIprGpu {
    /// Creates a new batch IPR pipeline for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BatchIpr BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BatchIpr Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader(&WGSL_BATCH_IPR, Some("BatchIpr Shader"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BatchIpr Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("batch_ipr"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute IPR for `n_vectors` eigenvectors, each of dimension `dim`.
    ///
    /// `eigenvectors_buf` layout: `[n_vectors × dim]` contiguous f32.
    /// Returns buffer of `[n_vectors]` f32 IPR values.
    pub fn dispatch(
        &self,
        eigenvectors_buf: &wgpu::Buffer,
        ipr_out_buf: &wgpu::Buffer,
        dim: u32,
        n_vectors: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = IprParams { dim, n_vectors };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchIpr Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BatchIpr BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: eigenvectors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ipr_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BatchIpr Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BatchIpr Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_vectors.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

use wgpu::util::DeviceExt;

#[cfg(all(test, feature = "gpu"))]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use std::sync::Arc;

    fn get_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available_sync()
    }

    #[tokio::test]
    async fn test_batch_ipr_creation() {
        let Some(device) = get_device() else {
            return;
        };
        let _batch_ipr = BatchIprGpu::new(device);
    }

    #[tokio::test]
    async fn test_batch_ipr_uniform_vector() {
        let Some(device) = get_device() else {
            return;
        };
        let dim = 64u32;
        let n_vectors = 4u32;
        // Uniform normalized vector: each component = 1/sqrt(dim)
        // IPR = Σ|ψ_i|⁴ = dim * (1/dim)² = 1/dim
        let inv_sqrt_dim = 1.0 / (dim as f32).sqrt();
        let mut eigenvectors = vec![0.0f32; (dim * n_vectors) as usize];
        for v in eigenvectors.chunks_mut(dim as usize) {
            v.fill(inv_sqrt_dim);
        }

        let batch_ipr = BatchIprGpu::new(device.clone());
        let eig_buf = device.create_buffer_f32_init("batch_ipr:eig", &eigenvectors);
        let ipr_out_buf = device
            .create_f32_rw_buffer("batch_ipr:out", n_vectors as usize)
            .unwrap();

        batch_ipr.dispatch(&eig_buf, &ipr_out_buf, dim, n_vectors);

        let ipr_values = device
            .read_buffer_f32(&ipr_out_buf, n_vectors as usize)
            .unwrap();
        let expected_ipr = 1.0 / dim as f32;
        for (i, &ipr) in ipr_values.iter().enumerate() {
            assert!(
                (ipr - expected_ipr).abs() < 1e-5,
                "Vector {i}: IPR {ipr} should be ≈ {expected_ipr} (uniform)"
            );
        }
    }

    #[tokio::test]
    async fn test_batch_ipr_localized_vector() {
        let Some(device) = get_device() else {
            return;
        };
        let dim = 32u32;
        let n_vectors = 4u32;
        // One-hot vectors: IPR = 1.0 (fully localized)
        let mut eigenvectors = vec![0.0f32; (dim * n_vectors) as usize];
        for (vec_idx, v) in eigenvectors.chunks_mut(dim as usize).enumerate() {
            v[vec_idx % dim as usize] = 1.0;
        }

        let batch_ipr = BatchIprGpu::new(device.clone());
        let eig_buf = device.create_buffer_f32_init("batch_ipr:eig", &eigenvectors);
        let ipr_out_buf = device
            .create_f32_rw_buffer("batch_ipr:out", n_vectors as usize)
            .unwrap();

        batch_ipr.dispatch(&eig_buf, &ipr_out_buf, dim, n_vectors);

        let ipr_values = device
            .read_buffer_f32(&ipr_out_buf, n_vectors as usize)
            .unwrap();
        for (i, &ipr) in ipr_values.iter().enumerate() {
            assert!(
                (ipr - 1.0).abs() < 1e-5,
                "Vector {i}: IPR {ipr} should be ≈ 1.0 (localized)"
            );
        }
    }
}
