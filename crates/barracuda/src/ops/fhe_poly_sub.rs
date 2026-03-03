//! FHE Polynomial Subtraction Operation
//!
//! **Purpose**: Subtract two FHE ciphertext polynomials on GPU
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (modular subtraction)
//! - ✅ Production-ready (full error handling)
//! - ✅ Canonical pattern: Tensor inputs/outputs, device from runtime

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// FHE polynomial subtraction operation
///
/// Subtracts two polynomials coefficient-wise with modular reduction.
///
/// ## Mathematical Operation
///
/// Given polynomials a(X) and b(X) over Z_q[X]/(X^N + 1):
/// ```text
/// result(X) = a(X) - b(X) mod q
/// ```
///
/// Where each coefficient is reduced modulo q.
pub struct FhePolySub {
    poly_a: Tensor,
    poly_b: Tensor,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    degree: u32,
    modulus: u64,
}

impl FhePolySub {
    /// Create a new FHE polynomial subtraction operation
    ///
    /// ## Parameters
    ///
    /// - `poly_a`: First polynomial tensor (u32 pairs representing u64 coefficients)
    /// - `poly_b`: Second polynomial tensor (u32 pairs representing u64 coefficients)
    /// - `degree`: Polynomial degree (N), typically 2048, 4096, or 8192
    /// - `modulus`: Modulus q (large prime, e.g., 2^60)
    pub fn new(poly_a: Tensor, poly_b: Tensor, degree: u32, modulus: u64) -> Result<Self> {
        // Validate inputs
        let expected_size = (degree as usize) * 2; // u32 pairs for u64
        if poly_a.len() != expected_size {
            return Err(BarracudaError::Device(format!(
                "poly_a length {} doesn't match expected {} (degree {} * 2)",
                poly_a.len(),
                expected_size,
                degree
            )));
        }
        if poly_b.len() != expected_size {
            return Err(BarracudaError::Device(format!(
                "poly_b length {} doesn't match expected {} (degree {} * 2)",
                poly_b.len(),
                expected_size,
                degree
            )));
        }

        // Ensure both tensors are on same device
        if !std::ptr::eq(poly_a.device().as_ref(), poly_b.device().as_ref()) {
            return Err(BarracudaError::Device(
                "poly_a and poly_b must be on the same device".to_string(),
            ));
        }

        if modulus == 0 {
            return Err(BarracudaError::Device(
                "Modulus must be non-zero".to_string(),
            ));
        }

        let device = poly_a.device();

        // Load WGSL shader
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("FHE Polynomial Subtraction Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("fhe_poly_sub.wgsl").into()),
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FHE Poly Sub Bind Group Layout"),
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FHE Poly Sub Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FHE Poly Sub Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "fhe_poly_sub",
                cache: None,
                compilation_options: Default::default(),
            });

        Ok(Self {
            poly_a,
            poly_b,
            pipeline,
            bind_group_layout,
            degree,
            modulus,
        })
    }

    /// Execute polynomial subtraction on GPU
    ///
    /// ## Returns
    ///
    /// Result tensor: (poly_a - poly_b) mod q
    /// Data stays on GPU (no CPU readback)
    pub fn execute(self) -> Result<Tensor> {
        let device = self.poly_a.device();

        // Create output buffer (u32 pairs for u64 coefficients)
        let result_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FHE Result Buffer"),
            size: (self.degree as u64 * 2 * std::mem::size_of::<u32>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            degree: u32,
            modulus_lo: u32,
            modulus_hi: u32,
            _padding: [u32; 5],
        }

        let params = Params {
            degree: self.degree,
            modulus_lo: self.modulus as u32,
            modulus_hi: (self.modulus >> 32) as u32,
            _padding: [0; 5],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FHE Params Buffer"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FHE Poly Sub Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.poly_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.poly_b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute shader
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FHE Poly Sub Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FHE Poly Sub Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::FHE);
            let workgroups = self.degree.div_ceil(optimal_wg_size);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return tensor (data stays on GPU)
        Ok(Tensor::from_buffer(
            result_buffer,
            vec![self.degree as usize * 2], // u32 pairs
            device.clone(),
        ))
    }
}

/// Helper: Create FHE polynomial tensor from u64 coefficients
pub async fn create_fhe_poly_tensor(
    poly: &[u64],
    device: Arc<crate::device::WgpuDevice>,
) -> Result<Tensor> {
    let poly_u32: Vec<u32> = poly
        .iter()
        .flat_map(|&val| vec![val as u32, (val >> 32) as u32])
        .collect();
    Tensor::from_data_pod(&poly_u32, vec![poly_u32.len()], device)
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    use crate::ops::fhe_poly_add::create_fhe_poly_tensor;

    #[allow(unused_imports)]
    use wgpu::util::DeviceExt;

    #[tokio::test]
    async fn test_fhe_poly_sub_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let degree = 8;
        let modulus = 97;

        let poly_a_data = vec![50u64, 60, 70, 80, 90, 85, 75, 65];
        let poly_b_data = vec![10u64, 20, 30, 40, 50, 60, 70, 80];

        let poly_a = create_fhe_poly_tensor(&poly_a_data, device.clone())
            .await
            .unwrap();
        let poly_b = create_fhe_poly_tensor(&poly_b_data, device.clone())
            .await
            .unwrap();

        let op = FhePolySub::new(poly_a, poly_b, degree, modulus).unwrap();
        let result_tensor = op.execute().unwrap();

        // Read back for testing
        let size = result_tensor.len();
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Staging"),
            size: (size * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            result_tensor.buffer(),
            0,
            &staging_buffer,
            0,
            (size * std::mem::size_of::<u32>()) as u64,
        );
        device.submit_and_poll(Some(encoder.finish()));

        let result_u32: Vec<u32> = device.map_staging_buffer(&staging_buffer, size).unwrap();

        let result: Vec<u64> = result_u32
            .chunks(2)
            .map(|pair| (pair[0] as u64) | ((pair[1] as u64) << 32))
            .collect();

        let expected: Vec<u64> = vec![40, 40, 40, 40, 40, 25, 5, 82];
        assert_eq!(result, expected);
    }
}
