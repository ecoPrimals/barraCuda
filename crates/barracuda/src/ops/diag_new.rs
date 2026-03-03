//! Diag - Extract matrix diagonal - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Extracts the main diagonal from a 2D matrix:
//! ```text
//! Input:  [[1, 2, 3],
//!          [4, 5, 6],
//!          [7, 8, 9]]
//! Output: [1, 5, 9]
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/diag_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

pub struct Diag {
    input: Tensor,
}

impl Diag {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }
    
    fn wgsl_shader() -> &'static str {
        &*SHADER_F32
    }
    
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        
        // For now, only support 2D matrices
        if shape.len() != 2 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0],  // placeholder for "2D"
                actual: shape.to_vec(),
            });
        }
        
        let rows = shape[0];
        let cols = shape[1];
        let diag_size = rows.min(cols);
        
        let output_buffer = device.create_buffer_f32(diag_size)?;
        
        // Create params buffer
        let params_data = [
            rows as u32,
            cols as u32,
            diag_size as u32,
        ];
        let params_buffer = device.create_uniform_buffer(&params_data)?;
        
        let bind_group_layout = device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Diag BGL"),
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
            }
        );
        
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Diag BG"),
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
        
        let shader = device.compile_shader(Self::wgsl_shader(), Some("Diag"));
        let pipeline_layout = device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Diag PL"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );
        
        let pipeline = device.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Diag Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
            }
        );
        
        let mut encoder = device.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Diag Encoder"),
            }
        );
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Diag Pass"),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(&device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (diag_size as u32 + optimal_wg_size - 1) / optimal_wg_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        device.submit_and_poll(Some(encoder.finish()));
        
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![diag_size],
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn diag_wgsl(self) -> Result<Self> {
        Diag::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_diag_square_matrix() {
        let Some(device) = get_test_device_if_gpu_available().await else { return };
        // 3x3 matrix
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::from_vec_on(input_data, vec![3, 3], device)
            .await
            .unwrap();
        
        let result = input.diag_wgsl().unwrap().to_vec().unwrap();
        assert_eq!(result, vec![1.0, 5.0, 9.0]);
    }

    #[tokio::test]
    async fn test_diag_rectangular() {
        let Some(device) = get_test_device_if_gpu_available().await else { return };
        // 2x3 matrix
        let input_data = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let input = Tensor::from_vec_on(input_data, vec![2, 3], device)
            .await
            .unwrap();
        
        let result = input.diag_wgsl().unwrap().to_vec().unwrap();
        assert_eq!(result, vec![1.0, 5.0]);
    }
}
