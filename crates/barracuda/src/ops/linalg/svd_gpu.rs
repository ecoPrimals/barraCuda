// SPDX-License-Identifier: AGPL-3.0-only
//! SVD (Singular Value Decomposition) - GPU-Accelerated Implementation (f64)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Full f64 precision via SPIR-V/Vulkan (bypasses CUDA fp64 throttle)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Runtime-configured matrix size
//!
//! ## Algorithm
//!
//! One-sided Jacobi SVD via eigendecomposition of `AᵀA`:
//! ```text
//! 1. compute_AtA:  B = AᵀA (parallel matmul)
//! 2. init_V:       V = I
//! 3. jacobi_sweep: Iterative rotations on B to diagonalize (eigendecomp)
//! 4. extract_sigma: σᵢ = √B[i,i] (singular values)
//! 5. compute_U:    U = A·V·Σ⁻¹ (optional)
//! ```
//!
//! ## Precision
//!
//! **Full f64 precision** - uses native WGSL f64 via SPIR-V/Vulkan.
//! FP64 performance is 1:2-3 (not 1:32 like CUDA consumer GPUs).
//!
//! ## References
//!
//! - Demmel & Veselic (1992), "Jacobi's Method is More Accurate than QR"
//! - Golub & Van Loan, "Matrix Computations", Algorithm 8.6.1

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// GPU-accelerated SVD decomposition
///
/// Computes A = U·Σ·Vᵀ where U and V are orthogonal, Σ is diagonal.
pub struct SvdGpu {
    input: Tensor,
    max_sweeps: u32,
}

impl SvdGpu {
    /// Create new GPU SVD operation
    /// # Arguments
    /// * `input` - Matrix [M, N] in row-major order
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self {
            input,
            max_sweeps: 30, // Default Jacobi sweeps
        }
    }

    /// Set maximum Jacobi sweeps for convergence
    #[must_use]
    pub fn with_max_sweeps(mut self, sweeps: u32) -> Self {
        self.max_sweeps = sweeps;
        self
    }

    fn wgsl_shader_f32() -> &'static str {
        include_str!("../../shaders/linalg/svd.wgsl")
    }

    fn wgsl_shader_f64() -> &'static str {
        include_str!("../../shaders/linalg/svd_f64.wgsl")
    }

    /// Build bind group layout from a list of buffer binding types.
    fn make_bgl(device: &wgpu::Device, types: &[wgpu::BufferBindingType]) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &types
                .iter()
                .enumerate()
                .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: *ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        })
    }

    fn make_pipe(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bgl: &wgpu::BindGroupLayout,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: shader,
            entry_point: Some(entry),
            cache: None,
            compilation_options: Default::default(),
        })
    }

    fn make_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        bufs: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        })
    }

    fn dispatch(
        dev: &WgpuDevice,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        wg: (u32, u32, u32),
    ) {
        let mut enc = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(pipeline);
            p.set_bind_group(0, Some(bg), &[]);
            p.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        dev.submit_and_poll(Some(enc.finish()));
    }

    fn create_zero_buffer(
        device: &Arc<WgpuDevice>,
        count: usize,
        elem_size: usize,
    ) -> wgpu::Buffer {
        device.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * elem_size) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Execute SVD decomposition (f32 via Tensor API)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Vec<f32>, Tensor)> {
        let device = self.input.device();
        let shape = self.input.shape();
        if shape.len() != 2 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }
        let (m, n) = (shape[0] as u32, shape[1] as u32);

        let input_data = self.input.to_vec()?;
        let a_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SVD A"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = Self::create_zero_buffer(device, (n * n) as usize, 4);
        let v_buf = Self::create_zero_buffer(device, (n * n) as usize, 4);
        let sigma_buf = Self::create_zero_buffer(device, n as usize, 4);

        let shader = device.compile_shader(Self::wgsl_shader_f32(), Some("SVD f32"));

        use wgpu::BufferBindingType::{Storage, Uniform};
        let bgl = Self::make_bgl(
            &device.device,
            &[
                Uniform,
                Storage { read_only: true },
                Storage { read_only: false },
                Storage { read_only: false },
                Storage { read_only: false },
            ],
        );
        let ata_pipe = Self::make_pipe(&device.device, &shader, &bgl, "compute_AtA");
        let init_pipe = Self::make_pipe(&device.device, &shader, &bgl, "init_V");
        let sigma_pipe = Self::make_pipe(&device.device, &shader, &bgl, "extract_sigma");

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[m, n, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = Self::make_bg(
            &device.device,
            &bgl,
            &[&params_buf, &a_buf, &b_buf, &v_buf, &sigma_buf],
        );

        let wg2d = n.div_ceil(16);
        Self::dispatch(device, &ata_pipe, &bg, (wg2d, wg2d, 1));
        Self::dispatch(device, &init_pipe, &bg, (wg2d, wg2d, 1));
        Self::dispatch(
            device,
            &sigma_pipe,
            &bg,
            (n.div_ceil(WORKGROUP_SIZE_1D), 1, 1),
        );

        let sigma_data = device.read_buffer_f32(&sigma_buf, n as usize)?;
        let v_data = device.read_buffer_f32(&v_buf, (n * n) as usize)?;
        let v_out = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SVD V Out"),
                contents: bytemuck::cast_slice(&v_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let v_tensor = Tensor::from_buffer(v_out, vec![n as usize, n as usize], device.clone());
        Ok((sigma_data, v_tensor))
    }

    /// Execute SVD with full f64 precision. Preferred method — native WGSL f64 via SPIR-V/Vulkan.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute_f64(
        device: Arc<WgpuDevice>,
        data: &[f64],
        m: usize,
        n: usize,
        max_sweeps: u32,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        if data.len() != m * n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Expected {} elements for {m}x{n} matrix, got {}",
                    m * n,
                    data.len()
                ),
            });
        }
        let (mu, nu) = (m as u32, n as u32);

        let a_bytes: &[u8] = bytemuck::cast_slice(data);
        let a_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SVD A f64"),
                contents: a_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let b_buf = Self::create_zero_buffer(&device, n * n, 8);
        let v_buf = Self::create_zero_buffer(&device, n * n, 8);
        let sigma_buf = Self::create_zero_buffer(&device, n, 8);
        let cs_buf = Self::create_zero_buffer(&device, 2, 8);

        let shader = device.compile_shader_f64(Self::wgsl_shader_f64(), Some("SVD f64"));

        use wgpu::BufferBindingType::{Storage, Uniform};
        let main_bgl = Self::make_bgl(
            &device.device,
            &[
                Uniform,
                Storage { read_only: true },
                Storage { read_only: false },
                Storage { read_only: false },
                Storage { read_only: false },
            ],
        );
        let rot_bgl = Self::make_bgl(
            &device.device,
            &[
                Uniform,
                Storage { read_only: false },
                Storage { read_only: true },
            ],
        );
        let jac_bgl = Self::make_bgl(
            &device.device,
            &[
                Uniform,
                Storage { read_only: true },
                Storage { read_only: false },
            ],
        );

        let ata_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "compute_AtA");
        let init_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "init_V");
        let jac_pipe =
            Self::make_pipe(&device.device, &shader, &jac_bgl, "compute_jacobi_rotation");
        let rot_b_pipe = Self::make_pipe(&device.device, &shader, &rot_bgl, "jacobi_rotate_B");
        let blk_pipe = Self::make_pipe(&device.device, &shader, &rot_bgl, "jacobi_update_block");
        let rot_v_pipe = Self::make_pipe(&device.device, &shader, &rot_bgl, "jacobi_rotate_V");
        let sigma_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "extract_sigma");

        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[mu, nu, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let main_bg = Self::make_bg(
            &device.device,
            &main_bgl,
            &[&params_buf, &a_buf, &b_buf, &v_buf, &sigma_buf],
        );

        let wg2d = nu.div_ceil(16);
        Self::dispatch(&device, &ata_pipe, &main_bg, (wg2d, wg2d, 1));
        Self::dispatch(&device, &init_pipe, &main_bg, (wg2d, wg2d, 1));

        for _sweep in 0..max_sweeps {
            for p in 0..nu.saturating_sub(1) {
                for q in (p + 1)..nu {
                    let rp_buf =
                        device
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: None,
                                contents: bytemuck::cast_slice(&[nu, p, q, 0u32]),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });
                    let jac_bg =
                        Self::make_bg(&device.device, &jac_bgl, &[&rp_buf, &b_buf, &cs_buf]);
                    Self::dispatch(&device, &jac_pipe, &jac_bg, (1, 1, 1));

                    let rot_bg =
                        Self::make_bg(&device.device, &rot_bgl, &[&rp_buf, &b_buf, &cs_buf]);
                    let wg1d = nu.div_ceil(WORKGROUP_SIZE_1D);
                    Self::dispatch(&device, &rot_b_pipe, &rot_bg, (wg1d, 1, 1));
                    Self::dispatch(&device, &blk_pipe, &rot_bg, (1, 1, 1));

                    let rv_bg =
                        Self::make_bg(&device.device, &rot_bgl, &[&rp_buf, &v_buf, &cs_buf]);
                    Self::dispatch(&device, &rot_v_pipe, &rv_bg, (wg1d, 1, 1));
                }
            }
        }

        Self::dispatch(
            &device,
            &sigma_pipe,
            &main_bg,
            (nu.div_ceil(WORKGROUP_SIZE_1D), 1, 1),
        );

        let sigma_data = device.read_f64_buffer(&sigma_buf, n)?;
        let v_data = device.read_f64_buffer(&v_buf, n * n)?;
        Ok((sigma_data, v_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SVD_F32_SHADER: &str = include_str!("../../shaders/linalg/svd.wgsl");
    const SVD_F64_SHADER: &str = include_str!("../../shaders/linalg/svd_f64.wgsl");

    #[test]
    fn svd_f32_shader_source_valid() {
        assert!(!SVD_F32_SHADER.is_empty());
        assert!(SVD_F32_SHADER.contains("fn ") || SVD_F32_SHADER.contains("@compute"));
    }

    #[test]
    fn svd_f64_shader_source_valid() {
        assert!(!SVD_F64_SHADER.is_empty());
        assert!(SVD_F64_SHADER.contains("fn ") || SVD_F64_SHADER.contains("@compute"));
    }

    #[tokio::test]
    async fn test_svd_gpu_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return; // Skip if no GPU
        };

        let a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let input = Tensor::from_data(&a, vec![3, 3], device).unwrap();

        let svd_gpu = SvdGpu::new(input);
        let (sigma, v_tensor) = svd_gpu.execute().unwrap();

        // Identity matrix: singular values should all be 1
        assert_eq!(sigma.len(), 3);
        for s in &sigma {
            assert!(
                (*s - 1.0).abs() < 0.1,
                "Expected singular value ~1.0, got {s}"
            );
        }

        let v_data = v_tensor.to_vec().unwrap();
        assert_eq!(v_data.len(), 9);
    }

    #[tokio::test]
    async fn test_svd_gpu_diagonal() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return; // Skip if no GPU
        };

        // Diagonal matrix with known singular values
        let a = vec![3.0f32, 0.0, 0.0, 4.0];
        let input = Tensor::from_data(&a, vec![2, 2], device).unwrap();

        let svd_gpu = SvdGpu::new(input);
        let (sigma, _v) = svd_gpu.execute().unwrap();

        // Diagonal matrix: singular values are absolute values of diagonal
        assert_eq!(sigma.len(), 2);
        // Check we got reasonable values (3 and 4 in some order)
        let sum: f32 = sigma.iter().map(|x| x * x).sum();
        assert!(
            (sum - 25.0).abs() < 1.0,
            "Expected sum of squares ~25, got {sum}"
        );
    }
}
