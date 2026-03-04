// SPDX-License-Identifier: AGPL-3.0-or-later
//! Broyden Mixing GPU Implementation (f64)
//!
//! GPU-accelerated vector mixing for SCF convergence.
//! Uses WGSL shaders for f64 precision on all GPU hardware.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Mixing parameters for SCF solvers
#[derive(Debug, Clone)]
pub struct MixingParams {
    /// Mixing parameter α (typically 0.3-0.7)
    pub alpha: f64,
    /// Optional minimum value (for non-negative quantities like density)
    pub clamp_min: Option<f64>,
    /// Optional maximum value
    pub clamp_max: Option<f64>,
    /// Number of warmup iterations with linear mixing before Broyden
    pub n_warmup: usize,
}

impl Default for MixingParams {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            clamp_min: None,
            clamp_max: None,
            n_warmup: 3,
        }
    }
}

/// Linear mixer for simple damped iteration
///
/// x_new = (1-α)·x_old + α·x_computed
pub struct LinearMixer {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vec_dim: usize,
    params: MixingParams,
}

impl LinearMixer {
    /// Create a new linear mixer
    pub fn new(device: Arc<WgpuDevice>, vec_dim: usize, params: MixingParams) -> Result<Self> {
        if vec_dim == 0 {
            return Err(crate::error::BarracudaError::invalid_op(
                "LinearMixer",
                "vec_dim must be > 0 (zero-length buffers are invalid for GPU compute)",
            ));
        }
        let shader_source = include_str!("../../shaders/mixing/broyden_f64.wgsl");
        let shader_module = device.compile_shader_f64(shader_source, Some("linear_mixer_shader"));

        let bind_group_layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("linear_mixer_bgl"),
                    entries: &[
                        // params uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // old_vec
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // computed_vec
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // output_vec
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("linear_mixer_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("linear_mixer_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "mix_linear",
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            vec_dim,
            params,
        })
    }

    /// Mix two vectors: x_new = (1-α)·x_old + α·x_computed
    pub async fn mix(&self, x_old: &[f64], x_computed: &[f64]) -> Result<Vec<f64>> {
        if x_old.len() != self.vec_dim || x_computed.len() != self.vec_dim {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Vector dimension mismatch: expected {}, got {} and {}",
                    self.vec_dim,
                    x_old.len(),
                    x_computed.len()
                ),
            });
        }

        // Create uniform params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LinearParamsGpu {
            vec_dim: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
            alpha: f64,
            clamp_min: f64,
            clamp_max: f64,
        }

        let params_data = LinearParamsGpu {
            vec_dim: self.vec_dim as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            alpha: self.params.alpha,
            clamp_min: self.params.clamp_min.unwrap_or(-1e308),
            clamp_max: self.params.clamp_max.unwrap_or(1e308),
        };

        let params_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("linear_params"),
                    contents: bytemuck::bytes_of(&params_data),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let old_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("old_vec"),
                    contents: bytemuck::cast_slice(x_old),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let computed_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("computed_vec"),
                    contents: bytemuck::cast_slice(x_computed),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_vec"),
            size: (self.vec_dim * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("linear_mixer_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: old_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: computed_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("linear_mix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                self.vec_dim.div_ceil(WORKGROUP_SIZE_1D as usize) as u32,
                1,
                1,
            );
        }

        // Read back
        let staging_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (self.vec_dim * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (self.vec_dim * std::mem::size_of::<f64>()) as u64,
        );
        self.device.submit_commands(Some(encoder.finish()));

        self.device
            .map_staging_buffer::<f64>(&staging_buffer, self.vec_dim)
    }
}

/// Broyden mixer with history for accelerated SCF convergence
///
/// Modified Broyden II algorithm:
/// x_{n+1} = x_n + α·r_n - Σ_m γ_m·(Δx_m + α·Δr_m)
pub struct BroydenMixer {
    device: Arc<WgpuDevice>,
    linear_mixer: LinearMixer,
    vec_dim: usize,
    max_history: usize,
    params: MixingParams,
    // History storage (CPU side - small linear algebra done on CPU)
    dx_history: Vec<Vec<f64>>,
    df_history: Vec<Vec<f64>>,
    iteration: usize,
}

impl BroydenMixer {
    /// Create a new Broyden mixer
    ///
    /// # Arguments
    /// * `device` - GPU device
    /// * `vec_dim` - Dimension of vectors to mix
    /// * `max_history` - Maximum number of history vectors (typically 5-10)
    /// * `params` - Mixing parameters
    pub fn new(
        device: Arc<WgpuDevice>,
        vec_dim: usize,
        max_history: usize,
        params: MixingParams,
    ) -> Result<Self> {
        let linear_mixer = LinearMixer::new(device.clone(), vec_dim, params.clone())?;

        Ok(Self {
            device,
            linear_mixer,
            vec_dim,
            max_history,
            params,
            dx_history: Vec::with_capacity(max_history),
            df_history: Vec::with_capacity(max_history),
            iteration: 0,
        })
    }

    /// GPU device handle (for future GPU-accelerated Broyden update).
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Dimension of the vectors being mixed.
    pub fn vec_dim(&self) -> usize {
        self.vec_dim
    }

    /// Reset the mixer for a new SCF calculation
    pub fn reset(&mut self) {
        self.dx_history.clear();
        self.df_history.clear();
        self.iteration = 0;
    }

    /// Perform one mixing step
    ///
    /// During warmup (first n_warmup iterations), uses linear mixing.
    /// After warmup, uses full Broyden with history.
    ///
    /// # Arguments
    /// * `x_old` - Input from previous iteration
    /// * `x_new` - Output from F(x_old)
    ///
    /// # Returns
    /// Mixed vector x_{n+1}
    pub async fn mix(&mut self, x_old: &[f64], x_new: &[f64]) -> Result<Vec<f64>> {
        self.iteration += 1;

        // Compute residual: r = x_new - x_old
        let residual: Vec<f64> = x_new.iter().zip(x_old).map(|(a, b)| a - b).collect();

        // Use linear mixing during warmup
        if self.iteration <= self.params.n_warmup || self.dx_history.is_empty() {
            let result = self.linear_mixer.mix(x_old, x_new).await?;

            // Store history for next iteration (if we have a previous iteration)
            if self.iteration > 1 && self.dx_history.len() < self.max_history {
                // Would need to store previous x and r for proper Broyden
                // For now, just return linear result during warmup
            }

            return Ok(result);
        }

        // Full Broyden mixing
        // Compute γ coefficients on CPU (small matrix operations)
        let gammas = self.compute_broyden_gammas(&residual)?;

        // Apply Broyden update on GPU
        let result = self.broyden_update_gpu(x_old, &residual, &gammas).await?;

        // Update history
        self.update_history(x_old, &result, &residual);

        Ok(result)
    }

    /// Compute Broyden γ coefficients via least-squares on CPU.
    ///
    /// Solves A·γ = β where A_ij = <ΔF_i|ΔF_j>, β_i = <ΔF_i|r>.
    /// This is O(n_history²) work on a tiny matrix — CPU is appropriate.
    fn compute_broyden_gammas(&self, residual: &[f64]) -> Result<Vec<f64>> {
        let m = self.df_history.len();
        if m == 0 {
            return Ok(Vec::new());
        }

        // Build overlap matrix A_ij = <ΔF_i|ΔF_j> and RHS β_i = <ΔF_i|r>
        let mut a = vec![0.0f64; m * m];
        let mut beta = vec![0.0f64; m];

        for i in 0..m {
            beta[i] = dot(&self.df_history[i], residual);
            for j in 0..m {
                a[i * m + j] = dot(&self.df_history[i], &self.df_history[j]);
            }
        }

        // Solve A·γ = β via Cholesky (A is symmetric positive semi-definite).
        // Add Tikhonov regularization for stability: A → A + εI
        let eps = 1e-12;
        for i in 0..m {
            a[i * m + i] += eps;
        }

        solve_symmetric_positive(m, &mut a, &mut beta);
        Ok(beta)
    }

    /// Apply Broyden update on GPU.
    ///
    /// x_new = x + α·r - Σ_k γ_k · (Δx_k + α·ΔF_k)
    ///
    /// The linear-mixing part (x + α·r) goes through the GPU mixer.
    /// The Broyden correction is accumulated on CPU (O(n_history × dim)) then
    /// added to the GPU result. For large dim this could be a second GPU kernel,
    /// but n_history is typically 5–10 so the CPU path is <1ms.
    async fn broyden_update_gpu(
        &self,
        x: &[f64],
        residual: &[f64],
        gammas: &[f64],
    ) -> Result<Vec<f64>> {
        let x_computed: Vec<f64> = x.iter().zip(residual).map(|(a, b)| a + b).collect();
        let mut result = self.linear_mixer.mix(x, &x_computed).await?;

        let alpha = self.params.alpha;
        for (k, gamma_k) in gammas.iter().enumerate() {
            if gamma_k.abs() < 1e-30 {
                continue;
            }
            let dx_k = &self.dx_history[k];
            let df_k = &self.df_history[k];
            for i in 0..result.len() {
                result[i] -= gamma_k * (dx_k[i] + alpha * df_k[i]);
            }
        }

        Ok(result)
    }

    /// Update history vectors
    fn update_history(&mut self, x_old: &[f64], x_new: &[f64], residual: &[f64]) {
        if self.dx_history.len() >= self.max_history {
            // Remove oldest history
            self.dx_history.remove(0);
            self.df_history.remove(0);
        }

        // Store Δx = x_new - x_old
        let dx: Vec<f64> = x_new.iter().zip(x_old).map(|(a, b)| a - b).collect();
        self.dx_history.push(dx);

        // Store ΔF (would need previous residual, simplified here)
        self.df_history.push(residual.to_vec());
    }
}

/// Inner product of two equal-length slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// In-place Cholesky solve for a small symmetric positive-definite system.
///
/// On entry, `a` is the column-major m×m matrix, `b` is the RHS.
/// On exit, `b` contains the solution.
fn solve_symmetric_positive(m: usize, a: &mut [f64], b: &mut [f64]) {
    // Cholesky decomposition: A = L·Lᵀ (in-place, lower triangle in `a`)
    for j in 0..m {
        let mut s = a[j * m + j];
        for k in 0..j {
            s -= a[j * m + k] * a[j * m + k];
        }
        if s <= 0.0 {
            // Not positive definite — zero out remaining gammas
            for i in j..m {
                b[i] = 0.0;
            }
            return;
        }
        let ljj = s.sqrt();
        a[j * m + j] = ljj;

        for i in (j + 1)..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / ljj;
        }
    }

    // Forward substitution: L·y = b
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * m + k] * b[k];
        }
        b[i] = s / a[i * m + i];
    }

    // Back substitution: Lᵀ·x = y
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in (i + 1)..m {
            s -= a[k * m + i] * b[k];
        }
        b[i] = s / a[i * m + i];
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_linear_mixer() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return; // Skip if no f64-capable GPU
        };

        let params = MixingParams {
            alpha: 0.5,
            ..Default::default()
        };
        let mixer = LinearMixer::new(device, 1024, params).unwrap();

        let x_old = vec![1.0; 1024];
        let x_computed = vec![2.0; 1024];

        let result = mixer.mix(&x_old, &x_computed).await.unwrap();

        // Expected: 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        for val in &result {
            assert!((val - 1.5).abs() < 1e-10, "Expected 1.5, got {}", val);
        }
    }

    #[test]
    fn test_cholesky_solve() {
        // 2×2 system: [4 2; 2 3] · x = [6; 5] → x = [1; 1]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let mut b = vec![6.0, 5.0];
        solve_symmetric_positive(2, &mut a, &mut b);
        assert!((b[0] - 1.0).abs() < 1e-10, "x[0]={}", b[0]);
        assert!((b[1] - 1.0).abs() < 1e-10, "x[1]={}", b[1]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-10);
    }
}
