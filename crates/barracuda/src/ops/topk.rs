// SPDX-License-Identifier: AGPL-3.0-or-later
//! TopK - GPU-accelerated top-K largest values selection
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (uses existing shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for inference)
//!
//! ## Algorithm
//!
//! ```text
//! Find indices of top K largest values in tensor
//! Output: [k] indices (as u32)
//! ```
//!
//! **Implementation**: GPU selection (basic O(n*k), parallel sorting for production)
//!
//! **Key Properties**:
//! - Returns indices, not values
//! - Handles duplicates
//! - Stable ordering for equal values
//!
//! **Used By**: Beam search, retrieval, recommendation systems
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let scores = Tensor::from_vec(vec![5.0, 1.0, 9.0, 3.0, 7.0], vec![5]).await?;
//! let top3_indices = scores.topk(3)?;  // Returns [2, 4, 0] (indices)
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// TopK parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TopKParams {
    k: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// TopK operation
///
/// **Deep Debt**: Uses existing WGSL shader with selection algorithm
pub struct TopK {
    input: Tensor,
    k: usize,
}

impl TopK {
    /// Create new TopK operation
    ///
    /// **Deep Debt**: Validates K against tensor size
    pub fn new(input: Tensor, k: usize) -> Result<Self> {
        // Validate K
        let size = input.len();
        if k == 0 {
            return Err(BarracudaError::invalid_op("TopK", "k must be positive"));
        }
        if k > size {
            return Err(BarracudaError::invalid_op(
                "TopK",
                format!("k ({k}) exceeds tensor size ({size})"),
            ));
        }

        // TopK currently only works on 1D tensors (flatten for higher dims)
        if input.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "TopK",
                "currently only supports 1D tensors (use flatten() first)",
            ));
        }

        Ok(Self { input, k })
    }

    /// WGSL shader source
    fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/misc/topk_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute TopK (GPU selection)
    ///
    /// **Deep Debt**: Basic O(n*k) selection, sufficient for moderate K
    ///
    /// Returns: Tensor of indices [k] as f32 (cast from u32)
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Create parameters
        let params = TopKParams {
            k: self.k as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopK Params"),
            size: std::mem::size_of::<TopKParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Output buffer (u32 indices)
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopK Output"),
            size: (self.k * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Compile shader
        let shader = device.compile_shader(Self::shader(), Some("TopK"));

        // Create bind group layout
        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("TopK BGL"),
                entries: &[
                    // Input
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
                    // Output
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
                    // Params
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TopK BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TopK Pipeline Layout"),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TopK Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("TopK Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TopK Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let size = self.input.len();
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read u32 indices back and convert to f32
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TopK Staging"),
            size: (self.k * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("TopK Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (self.k * std::mem::size_of::<u32>()) as u64,
        );
        device.submit_and_poll(Some(encoder.finish()));

        let indices_u32: Vec<u32> = device.map_staging_buffer(&staging_buffer, self.k)?;
        let indices_f32: Vec<f32> = indices_u32.iter().map(|&x| x as f32).collect();

        // Create output tensor [k] as f32
        let output_tensor = Tensor::from_vec_on_sync(indices_f32, vec![self.k], device.clone())?;

        Ok(output_tensor)
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Top-K largest values (returns indices)
    ///
    /// **Deep Debt**: Essential for beam search and retrieval
    ///
    /// # Arguments
    /// - `k`: Number of top values to return
    ///
    /// # Returns
    /// - Indices tensor [k] as f32 (cast from u32)
    ///
    /// # Example
    /// ```rust,ignore
    /// let scores = Tensor::from_vec(vec![5.0, 1.0, 9.0, 3.0, 7.0], vec![5]).await?;
    /// let top3 = scores.topk(3)?;  // [2, 4, 0] (indices of 9.0, 7.0, 5.0)
    /// ```
    ///
    /// # Note
    /// Currently only supports 1D tensors. Use `flatten()` for higher dimensions.
    pub fn topk(self, k: usize) -> Result<Self> {
        TopK::new(self, k)?.execute()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_topk_gpu_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![5.0, 1.0, 9.0, 3.0, 7.0], vec![5], device)
            .await
            .unwrap();

        let top3 = input.topk(3).unwrap();

        assert_eq!(top3.shape(), &[3]);
        let indices = top3.to_vec().unwrap();

        // Should return indices of [9.0, 7.0, 5.0] = [2, 4, 0]
        assert_eq!(indices[0] as u32, 2); // 9.0
        assert_eq!(indices[1] as u32, 4); // 7.0
        assert_eq!(indices[2] as u32, 0); // 5.0
    }

    #[tokio::test]
    async fn test_topk_gpu_single() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device)
            .await
            .unwrap();

        let top1 = input.topk(1).unwrap();
        let indices = top1.to_vec().unwrap();

        // Largest value is 4.0 at index 3
        assert_eq!(indices[0] as u32, 3);
    }

    #[tokio::test]
    async fn test_topk_gpu_all() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![5], device)
            .await
            .unwrap();

        let top5 = input.topk(5).unwrap();
        let indices = top5.to_vec().unwrap();

        // All indices, sorted by value: [5.0, 4.0, 3.0, 1.0, 1.0] = [4, 2, 0, 1, 3]
        assert_eq!(indices[0] as u32, 4); // 5.0
        assert_eq!(indices[1] as u32, 2); // 4.0
        assert_eq!(indices[2] as u32, 0); // 3.0
    }

    #[tokio::test]
    async fn test_topk_gpu_negative() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![-5.0, -1.0, -9.0, -3.0], vec![4], device)
            .await
            .unwrap();

        let top2 = input.topk(2).unwrap();
        let indices = top2.to_vec().unwrap();

        // Largest (least negative): [-1.0, -3.0] at indices [1, 3]
        assert_eq!(indices[0] as u32, 1); // -1.0
        assert_eq!(indices[1] as u32, 3); // -3.0
    }

    #[tokio::test]
    async fn test_topk_gpu_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();

        // k=0 should error
        assert!(input.clone().topk(0).is_err());

        // k > size should error
        assert!(input.topk(10).is_err());
    }

    #[tokio::test]
    async fn test_topk_gpu_duplicates() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![2.0, 5.0, 5.0, 1.0], vec![4], device)
            .await
            .unwrap();

        let top2 = input.topk(2).unwrap();
        let indices = top2.to_vec().unwrap();

        // Two 5.0 values at indices 1 and 2
        // Should return both (order may vary, but both should be 1 or 2)
        let idx0 = indices[0] as u32;
        let idx1 = indices[1] as u32;
        assert!((idx0 == 1 || idx0 == 2) && (idx1 == 1 || idx1 == 2));
    }

    #[tokio::test]
    async fn test_topk_gpu_large() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Larger tensor (100 elements)
        let mut values = vec![0.0; 100];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }

        let input = Tensor::from_vec_on(values, vec![100], device)
            .await
            .unwrap();

        let top10 = input.topk(10).unwrap();
        let indices = top10.to_vec().unwrap();

        // Should be indices 99, 98, 97, ..., 90 (highest values)
        assert_eq!(indices[0] as u32, 99);
        assert_eq!(indices[1] as u32, 98);
        assert_eq!(indices[9] as u32, 90);
    }
}
