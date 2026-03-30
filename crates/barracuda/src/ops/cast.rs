// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cast operation - Type conversion with multiple modes
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Cast mode for type conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastMode {
    /// f32 → f32 identity
    Identity = 0,
    /// f32 → i32 (truncate to integer, stored as f32)
    ToInt = 1,
    /// f32 → u32 (clamp to non-negative, truncate)
    ToUint = 2,
    /// Reinterpret bits as i32, convert to f32
    FromInt = 3,
    /// Reinterpret bits as u32, convert to f32
    FromUint = 4,
    /// f32 → f32 with clamp to [min, max]
    Clamp = 5,
    /// f32 → bool (0.0 or 1.0)
    ToBool = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CastParams {
    total: u32,
    mode: u32,
    min_val: f32,
    max_val: f32,
}

/// Type conversion with multiple modes (identity, f32↔int, clamp, bool).
pub struct Cast {
    input: Tensor,
    mode: CastMode,
    min_val: f32,
    max_val: f32,
}

impl Cast {
    /// Creates a cast with identity mode (no-op).
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self {
            input,
            mode: CastMode::Identity,
            min_val: f32::MIN,
            max_val: f32::MAX,
        }
    }

    /// Create a cast operation with a specific mode
    #[must_use]
    pub fn with_mode(input: Tensor, mode: CastMode) -> Self {
        Self {
            input,
            mode,
            min_val: f32::MIN,
            max_val: f32::MAX,
        }
    }

    /// Create a clamp cast operation
    #[must_use]
    pub fn with_clamp(input: Tensor, min_val: f32, max_val: f32) -> Self {
        Self {
            input,
            mode: CastMode::Clamp,
            min_val,
            max_val,
        }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> =
            std::sync::LazyLock::new(|| include_str!("../shaders/misc/cast_f64.wgsl").to_string());
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes the cast and returns the converted tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, GPU
    /// dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let params = CastParams {
            total: size as u32,
            mode: self.mode as u32,
            min_val: self.min_val,
            max_val: self.max_val,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cast Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Cast BGL"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cast BG"),
            layout: &bind_group_layout,
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Cast"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cast PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Cast Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Cast Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cast Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Cast tensor (identity mode for f32→f32 compatibility)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, GPU
    /// dispatch fails, or the device is lost.
    pub fn cast(self) -> Result<Self> {
        Cast::new(self).execute()
    }

    /// Cast tensor with a specific mode
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, GPU
    /// dispatch fails, or the device is lost.
    pub fn cast_mode(self, mode: CastMode) -> Result<Self> {
        Cast::with_mode(self, mode).execute()
    }

    /// Clamp tensor values to [min, max]
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, GPU
    /// dispatch fails, or the device is lost.
    pub fn cast_clamp(self, min_val: f32, max_val: f32) -> Result<Self> {
        Cast::with_clamp(self, min_val, max_val).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cast_cpu(x: f32) -> f32 {
        // Currently f32 -> f32 identity
        x
    }

    #[test]
    fn test_cast_mode_values() {
        assert_eq!(CastMode::Identity as u32, 0);
        assert_eq!(CastMode::ToInt as u32, 1);
        assert_eq!(CastMode::ToUint as u32, 2);
        assert_eq!(CastMode::FromInt as u32, 3);
        assert_eq!(CastMode::FromUint as u32, 4);
        assert_eq!(CastMode::Clamp as u32, 5);
        assert_eq!(CastMode::ToBool as u32, 6);
    }

    #[test]
    fn test_cast_params_layout() {
        let params = CastParams {
            total: 100,
            mode: 0,
            min_val: -1.0,
            max_val: 1.0,
        };
        assert_eq!(std::mem::size_of::<CastParams>(), 16);
        assert_eq!(params.total, 100);
    }

    #[tokio::test]
    async fn test_cast_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![1.0, 2.0, 3.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();

        let expected: Vec<f32> = input_data.iter().map(|&x| cast_cpu(x)).collect();
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_cast_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Negative values
        let input_data = vec![-10.0, -5.0, -1.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();
        for (r, orig) in result.iter().zip(input_data.iter()) {
            assert!((r - orig).abs() < 1e-6);
        }

        // Mixed positive/negative/zero
        let input_data = vec![-5.5, 0.0, 5.5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();
        for (r, orig) in result.iter().zip(input_data.iter()) {
            assert!((r - orig).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_cast_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Very small values
        let input_data = vec![1e-10, -1e-10, 1e-6, -1e-6];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();
        for (r, orig) in result.iter().zip(input_data.iter()) {
            assert!((r - orig).abs() < 1e-12);
        }

        // Large values
        let input_data = vec![1e10, -1e10, 1e6, -1e6];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();
        for (r, orig) in result.iter().zip(input_data.iter()) {
            assert!((r - orig).abs() < 1e4);
        }
    }

    #[tokio::test]
    async fn test_cast_large_tensor() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1000 elements
        let input_data: Vec<f32> = (0..1000).map(|i| (i as f32).mul_add(0.1, -50.0)).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![1000], device)
            .await
            .unwrap();

        let result = input.cast().unwrap().to_vec().unwrap();
        let expected: Vec<f32> = input_data.iter().map(|&x| cast_cpu(x)).collect();

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_cast_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test FP32 precision preservation
        let input_data = vec![-123.456, -78.901, -2.345, 0.0, 1.234, 56.789, 123.456];
        let input = Tensor::from_vec_on(input_data.clone(), vec![7], device)
            .await
            .unwrap();
        let result = input.cast().unwrap().to_vec().unwrap();
        let expected: Vec<f32> = input_data.iter().map(|&x| cast_cpu(x)).collect();

        // Verify FP32 precision (should be exact for f32->f32)
        let max_error = result
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(max_error < 1e-6, "Max error: {max_error} exceeds threshold");
    }
}
