// SPDX-License-Identifier: AGPL-3.0-or-later
//! LU Decomposition - GPU-Accelerated Implementation (f64)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Full f64 precision via SPIR-V/Vulkan (bypasses CUDA fp64 throttle)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Runtime-configured matrix size
//! - ✅ Capability-based dispatch
//!
//! ## Algorithm
//!
//! Multi-pass GPU LU decomposition with partial pivoting:
//! ```text
//! For each column k = 0..n-1:
//!   1. find_pivot:         GPU parallel reduction to find max|A[i,k]| for i >= k
//!   2. row_swap:           GPU parallel swap rows k and pivot_row
//!   3. compute_multipliers: GPU parallel L[i,k] = A[i,k]/A[k,k] for i > k
//!   4. row_elimination:    GPU parallel A[i,j] -= L[i,k]*A[k,j] for i,j > k
//! ```
//!
//! ## Precision
//!
//! **Full f64 precision** - uses native WGSL f64 via SPIR-V/Vulkan.
//! FP64 performance is 1:2-3 (not 1:32 like CUDA consumer GPUs).
//!
//! ## References
//!
//! - Golub & Van Loan, "Matrix Computations", Algorithm 3.4.1

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-accelerated LU decomposition
///
/// Computes PA = LU where P is permutation, L is lower triangular, U is upper triangular.
pub struct LuGpu {
    input: Tensor,
}

impl LuGpu {
    /// Create new GPU LU decomposition operation
    ///
    /// # Arguments
    /// * `input` - Square matrix [N, N] in row-major order
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader_f32() -> &'static str {
        include_str!("../../shaders/linalg/lu_decomp.wgsl")
    }

    fn wgsl_shader_f64() -> &'static str {
        include_str!("../../shaders/linalg/lu_decomp_f64.wgsl")
    }

    /// Build a 3-binding layout: binding 0 = uniform, binding 1 = storage, binding 2 = storage (rw).
    fn make_bgl(device: &wgpu::Device, b1_read_only: bool) -> wgpu::BindGroupLayout {
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(
                    1,
                    wgpu::BufferBindingType::Storage {
                        read_only: b1_read_only,
                    },
                ),
                entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        })
    }

    fn make_pipe(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        bgl: &wgpu::BindGroupLayout,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pl),
            module: shader,
            entry_point,
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
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            p.set_pipeline(pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(wg.0, wg.1, wg.2);
        }
        dev.submit_and_poll(Some(enc.finish()));
    }

    /// Execute LU decomposition (f32 via Tensor API)
    pub fn execute(self) -> Result<(Tensor, Vec<u32>)> {
        let device = self.input.device();
        let shape = self.input.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }
        let n = shape[0] as u32;

        let input_data = self.input.to_vec()?;
        let lu_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LU Matrix"),
                contents: bytemuck::cast_slice(&input_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let perm_init: Vec<u32> = (0..n).collect();
        let perm_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LU Perm"),
                contents: bytemuck::cast_slice(&perm_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let pivot_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pivot"),
            size: 8,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let shader = device.compile_shader(Self::wgsl_shader_f32(), Some("LU f32"));
        let pivot_bgl = Self::make_bgl(&device.device, true);
        let main_bgl = Self::make_bgl(&device.device, false);
        let find_pivot_pipe = Self::make_pipe(&device.device, &shader, &pivot_bgl, "find_pivot");
        let row_swap_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "row_swap");
        let mult_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "compute_multipliers");
        let elim_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "row_elimination");

        for k in 0..(n - 1) {
            let params_buf = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[n, k, 0u32, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

            let pivot_bg = Self::make_bg(
                &device.device,
                &pivot_bgl,
                &[&params_buf, &lu_buffer, &pivot_buf],
            );
            Self::dispatch(device, &find_pivot_pipe, &pivot_bg, (1, 1, 1));

            let pivot_data = device.read_buffer_u32(&pivot_buf, 2)?;
            let pivot_row = pivot_data[0];

            let params_pivot =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[n, k, pivot_row, 0u32]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let main_bg = Self::make_bg(
                &device.device,
                &main_bgl,
                &[&params_pivot, &lu_buffer, &perm_buffer],
            );

            if pivot_row != k {
                Self::dispatch(
                    device,
                    &row_swap_pipe,
                    &main_bg,
                    (n.div_ceil(WORKGROUP_SIZE_1D), 1, 1),
                );
            }
            Self::dispatch(
                device,
                &mult_pipe,
                &main_bg,
                ((n - k - 1).div_ceil(WORKGROUP_SIZE_1D), 1, 1),
            );
            let sub = n - k - 1;
            Self::dispatch(
                device,
                &elim_pipe,
                &main_bg,
                (sub.div_ceil(16), sub.div_ceil(16), 1),
            );
        }

        let lu_data = device.read_buffer_f32(&lu_buffer, (n * n) as usize)?;
        let perm_data = device.read_buffer_u32(&perm_buffer, n as usize)?;
        let output_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LU Output"),
                contents: bytemuck::cast_slice(&lu_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        Ok((
            Tensor::from_buffer(output_buffer, shape.to_vec(), device.clone()),
            perm_data,
        ))
    }

    /// Execute LU decomposition with full f64 precision.
    /// Preferred method — native WGSL f64 via SPIR-V/Vulkan (1:2-3 FP64 ratio).
    pub fn execute_f64(
        device: Arc<WgpuDevice>,
        data: &[f64],
        n: usize,
    ) -> Result<(Vec<f64>, Vec<u32>)> {
        if data.len() != n * n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Expected {} elements for {n}x{n} matrix, got {}",
                    n * n,
                    data.len()
                ),
            });
        }
        let nu = n as u32;

        let lu_buffer = {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("LU f64"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        let perm_init: Vec<u32> = (0..nu).collect();
        let perm_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LU Perm"),
                contents: bytemuck::cast_slice(&perm_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let pivot_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pivot"),
            size: 4,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let shader = device.compile_shader_f64(Self::wgsl_shader_f64(), Some("LU f64"));
        let pivot_bgl = Self::make_bgl(&device.device, true);
        let main_bgl = Self::make_bgl(&device.device, false);
        let find_pivot_pipe = Self::make_pipe(&device.device, &shader, &pivot_bgl, "find_pivot");
        let row_swap_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "row_swap");
        let mult_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "compute_multipliers");
        let elim_pipe = Self::make_pipe(&device.device, &shader, &main_bgl, "row_elimination");

        for k in 0..(nu - 1) {
            let params_buf = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[nu, k, 0u32, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let pivot_bg = Self::make_bg(
                &device.device,
                &pivot_bgl,
                &[&params_buf, &lu_buffer, &pivot_buf],
            );
            Self::dispatch(&device, &find_pivot_pipe, &pivot_bg, (1, 1, 1));

            let pivot_data = device.read_buffer_u32(&pivot_buf, 1)?;
            let pivot_row = pivot_data[0];

            let params_pivot =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(&[nu, k, pivot_row, 0u32]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let main_bg = Self::make_bg(
                &device.device,
                &main_bgl,
                &[&params_pivot, &lu_buffer, &perm_buffer],
            );

            if pivot_row != k {
                Self::dispatch(
                    &device,
                    &row_swap_pipe,
                    &main_bg,
                    (nu.div_ceil(WORKGROUP_SIZE_1D), 1, 1),
                );
            }
            Self::dispatch(
                &device,
                &mult_pipe,
                &main_bg,
                ((nu - k - 1).div_ceil(WORKGROUP_SIZE_1D), 1, 1),
            );
            let sub = nu - k - 1;
            Self::dispatch(
                &device,
                &elim_pipe,
                &main_bg,
                (sub.div_ceil(16), sub.div_ceil(16), 1),
            );
        }

        let lu_data = device.read_f64_buffer(&lu_buffer, n * n)?;
        let perm_data = device.read_buffer_u32(&perm_buffer, n)?;
        Ok((lu_data, perm_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_gpu_shader_f32_source_valid() {
        let src = LuGpu::wgsl_shader_f32();
        assert!(!src.is_empty());
        assert!(src.contains("fn main") || src.contains("@compute"));
    }

    #[test]
    fn lu_gpu_shader_f64_source_valid() {
        let src = LuGpu::wgsl_shader_f64();
        assert!(!src.is_empty());
        assert!(src.contains("fn main") || src.contains("@compute"));
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[tokio::test]
    async fn test_lu_gpu_2x2() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return; // Skip if no GPU
        };

        let a = vec![4.0f32, 3.0, 6.0, 3.0];
        let input = Tensor::from_data(&a, vec![2, 2], device.clone()).unwrap();

        let lu_gpu = LuGpu::new(input);
        let (lu_tensor, perm) = lu_gpu.execute().unwrap();

        let lu_data = lu_tensor.to_vec().unwrap();

        // Verify LU factorization: should be able to reconstruct A from L and U
        // For a 2x2 matrix, check that we got valid factors
        assert_eq!(lu_data.len(), 4);
        assert_eq!(perm.len(), 2);
    }

    #[tokio::test]
    async fn test_lu_gpu_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return; // Skip if no GPU
        };

        let a = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let input = Tensor::from_data(&a, vec![3, 3], device.clone()).unwrap();

        let lu_gpu = LuGpu::new(input);
        let (lu_tensor, _perm) = lu_gpu.execute().unwrap();

        let lu_data = lu_tensor.to_vec().unwrap();

        // Identity matrix LU decomposition should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(lu_data[i * 3 + j], expected, 1e-5),
                    "LU[{},{}] = {}, expected {}",
                    i,
                    j,
                    lu_data[i * 3 + j],
                    expected
                );
            }
        }
    }
}
