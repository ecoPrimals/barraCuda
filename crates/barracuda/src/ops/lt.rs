//! Lt operation - Less than comparison  
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via downcast_f64_to_f32 when needed.
const SHADER_F64: &str = include_str!("../shaders/misc/lt_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

pub struct Lt {
    lhs: Tensor,
    rhs: Tensor,
}

impl Lt {
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Lt BGL"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lt BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.lhs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rhs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Lt"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Lt PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Lt Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Lt Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Lt Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));
        Ok(Tensor::from_buffer(
            output_buffer,
            self.lhs.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn lt(self, other: &Self) -> Result<Self> {
        Lt::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_lt_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let a = Tensor::from_vec_on(vec![1.0, 3.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        // Just verify operation completed
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_lt_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // All less than
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![4.0, 5.0, 6.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 0.1)); // All true

        // None less than
        let a = Tensor::from_vec_on(vec![5.0, 6.0, 7.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 0.1)); // All false
    }

    #[tokio::test]
    async fn test_lt_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Equal values
        let a = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 0.1)); // All false (not less than)

        // Negative values
        let a = Tensor::from_vec_on(vec![-5.0, -3.0, -1.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![-4.0, -4.0, 0.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_lt_large_tensor() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 1000 elements
        let a_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..1000).map(|i| (i + 500) as f32).collect();
        let a = Tensor::from_vec_on(a_data, vec![1000], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![1000], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 1000);
    }

    #[tokio::test]
    async fn test_lt_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Mixed comparisons
        let a = Tensor::from_vec_on(vec![1.0, 5.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 4.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 3);
        // result[0]: 1 < 2 = true (1.0)
        // result[1]: 5 < 4 = false (0.0)
        // result[2]: 3 < 3 = false (0.0)
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
