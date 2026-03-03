//! Broadcast operation - Expand tensor dimensions with full NumPy-style broadcasting
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// f64 is the canonical source — f32 derived via downcast_f64_to_f32 when needed.
const SHADER_F64: &str = include_str!("../shaders/tensor/broadcast_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastParams {
    output_total: u32,
    ndim: u32,
    _padding: [u32; 2],
}

pub struct Broadcast {
    input: Tensor,
    target_shape: Vec<usize>,
}

impl Broadcast {
    pub fn new(input: Tensor, target_shape: Vec<usize>) -> Self {
        Self {
            input,
            target_shape,
        }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Compute input strides for broadcasting.
    /// A stride of 0 means the dimension is broadcast (size-1 dim).
    fn compute_input_strides(input_shape: &[usize], target_shape: &[usize]) -> Vec<u32> {
        let ndim = target_shape.len();
        let input_ndim = input_shape.len();

        // Pad input shape with leading 1s to match target ndim
        let mut padded_input: Vec<usize> = vec![1; ndim.saturating_sub(input_ndim)];
        padded_input.extend_from_slice(input_shape);

        // Compute regular strides for the input
        let mut strides = vec![0u32; ndim];
        let mut stride: u32 = 1;
        for d in (0..ndim).rev() {
            if padded_input[d] == 1 && target_shape[d] > 1 {
                // This dimension is broadcast: stride = 0
                strides[d] = 0;
            } else {
                strides[d] = stride;
            }
            // Only advance stride for non-broadcast dims
            if padded_input[d] > 1 {
                stride *= padded_input[d] as u32;
            }
        }

        strides
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let output_size: usize = self.target_shape.iter().product();
        let ndim = self.target_shape.len();

        // Compute broadcast strides
        let input_strides = Self::compute_input_strides(self.input.shape(), &self.target_shape);

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params uniform
        let params = BroadcastParams {
            output_total: output_size as u32,
            ndim: ndim as u32,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Broadcast Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create output_shape buffer
        let output_shape_u32: Vec<u32> = self.target_shape.iter().map(|&s| s as u32).collect();
        let output_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Broadcast Output Shape"),
                    contents: bytemuck::cast_slice(&output_shape_u32),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Create input_strides buffer
        let strides_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Broadcast Input Strides"),
                contents: bytemuck::cast_slice(&input_strides),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create shader module
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Broadcast Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Broadcast Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Broadcast Bind Group"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_shape_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: strides_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Broadcast Encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Broadcast Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.target_shape,
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Broadcast tensor to target shape using NumPy-style broadcasting rules.
    /// Dimensions of size 1 in the input are broadcast to match the target.
    /// # Arguments
    /// * `target_shape` - Target shape to broadcast to
    pub fn broadcast(self, target_shape: Vec<usize>) -> Result<Self> {
        Broadcast::new(self, target_shape).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[test]
    fn test_compute_input_strides_scalar_broadcast() {
        // [1] → [10]: stride 0 (broadcast)
        let strides = Broadcast::compute_input_strides(&[1], &[10]);
        assert_eq!(strides, vec![0]);
    }

    #[test]
    fn test_compute_input_strides_2d_broadcast() {
        // [3, 1] → [3, 4]: dim 0 normal, dim 1 broadcast
        let strides = Broadcast::compute_input_strides(&[3, 1], &[3, 4]);
        assert_eq!(strides, vec![1, 0]);
    }

    #[test]
    fn test_compute_input_strides_no_broadcast() {
        // [3, 4] → [3, 4]: no broadcast, normal strides
        let strides = Broadcast::compute_input_strides(&[3, 4], &[3, 4]);
        assert_eq!(strides, vec![4, 1]);
    }

    #[test]
    fn test_compute_input_strides_rank_expansion() {
        // [4] → [3, 4]: new dim 0 is broadcast, dim 1 is normal
        let strides = Broadcast::compute_input_strides(&[4], &[3, 4]);
        assert_eq!(strides, vec![0, 1]);
    }

    #[tokio::test]
    async fn test_broadcast_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Create scalar [5.0]
        let input_data = vec![5.0f32];
        let input = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();

        // Broadcast to shape [10]
        let result = input.broadcast(vec![10]).unwrap();
        let output = result.to_vec().unwrap();

        // All should be 5.0
        assert_eq!(output.len(), 10);
        for val in output.iter() {
            assert_eq!(*val, 5.0);
        }
    }

    #[tokio::test]
    async fn test_broadcast_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Broadcast single element to multiple
        let input_data = vec![9.0f32];
        let input = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();

        let result = input.broadcast(vec![5]).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 5);
        assert!(output.iter().all(|&x| x == 9.0));
    }

    #[tokio::test]
    async fn test_broadcast_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Small to large broadcast
        let input_data = vec![7.0f32];
        let input = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();

        let result = input.broadcast(vec![100]).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 100);
        assert!(output.iter().all(|&x| x == 7.0));
    }

    #[tokio::test]
    async fn test_broadcast_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Broadcast to large size
        let input_data = vec![2.78f32];
        let input = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();

        let result = input.broadcast(vec![1000]).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1000);
        assert!(output.iter().all(|&x| (x - 2.78).abs() < 1e-6));
    }

    #[tokio::test]
    async fn test_broadcast_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test determinism
        let input_data = vec![2.5f32];

        let input1 = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();
        let input2 = Tensor::from_data(&input_data, vec![1], device.clone()).unwrap();

        let result1 = input1.broadcast(vec![5]).unwrap();
        let result2 = input2.broadcast(vec![5]).unwrap();

        let output1 = result1.to_vec().unwrap();
        let output2 = result2.to_vec().unwrap();

        // Should be deterministic
        assert_eq!(output1, output2);
        assert_eq!(output1, vec![2.5, 2.5, 2.5, 2.5, 2.5]);
    }
}
