// SPDX-License-Identifier: AGPL-3.0-or-later
//! NAdam Optimizer - GPU-accelerated Nesterov Adam
//!
//! **Deep Debt Principles**:
//! - ✅ Reused existing WGSL shader (smart evolution!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready training)
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)
//!
//! ## Algorithm
//!
//! NAdam combines Nesterov momentum with Adam's adaptive learning:
//! ```text
//! m_t = β₁*m_{t-1} + (1-β₁)*g_t
//! v_t = β₂*v_{t-1} + (1-β₂)*g_t²
//! m̂_t = (β₁*m_t + (1-β₁)*g_t) / (1-β₁^t)  // Nesterov lookahead
//! v̂_t = v_t / (1-β₂^t)
//! θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
//! ```
//!
//! **Advantage**: Often faster convergence than vanilla Adam
//!
//! **Reference**: Dozat, 2016 - "Incorporating Nesterov Momentum into Adam"

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// NAdam parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NAdamParams {
    num_params: u32,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
    _padding: [u32; 2],
}

/// NAdam optimizer operation
pub struct NAdam {
    params: Tensor,
    gradients: Tensor,
    m: Option<Tensor>,
    v: Option<Tensor>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    step: usize,
}

impl NAdam {
    /// Create new NAdam optimizer step
    pub fn new(
        params: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        m: Option<Tensor>,
        v: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if params.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        Ok(Self {
            params,
            gradients,
            m,
            v,
            learning_rate,
            beta1,
            beta2,
            step,
        })
    }

    /// WGSL shader source
    fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!(
                    "../shaders/optimizer/nadam_f64.wgsl"
                ).to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute NAdam step (returns updated params, m, v)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.params.device();
        let size = self.params.len();

        // Create parameters
        let params = NAdamParams {
            num_params: size as u32,
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: 1e-8,
            step: self.step as u32,
            _padding: [0, 0],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NAdam Params"),
            size: std::mem::size_of::<NAdamParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create or get m/v buffers
        let zeros = vec![0.0f32; size];
        
        use wgpu::util::DeviceExt;
        let m_buffer = if let Some(ref m_tensor) = self.m {
            m_tensor.buffer().clone()
        } else {
            device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nadam_m_init"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };

        let v_buffer = if let Some(ref v_tensor) = self.v {
            v_tensor.buffer().clone()
        } else {
            device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nadam_v_init"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };

        // Output buffers
        let params_out_buffer = device.create_buffer_f32(size)?;
        let m_out_buffer = device.create_buffer_f32(size)?;
        let v_out_buffer = device.create_buffer_f32(size)?;

        // Compile shader
        let shader = device.compile_shader(Self::shader(), Some("NAdam"));

        // Create bind group layout
        let bgl = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("NAdam BGL"),
            entries: &[
                // Gradients (input)
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
                // Params (input)
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
                // M (input/output)
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
                // V (input/output)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params out
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // M out
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // V out
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
            label: Some("NAdam BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gradients.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: m_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: v_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("NAdam Pipeline Layout"),
            bind_group_layouts: &[&bgl],
                    immediate_size: 0,
        });

        let pipeline = device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NAdam Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
        });

        // Execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("NAdam Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NAdam Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            
            let caps = DeviceCapabilities::from_device(&device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32 + optimal_wg_size - 1) / optimal_wg_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Return updated tensors
        let params_shape = self.params.shape().to_vec();
        Ok((
            Tensor::from_buffer(params_out_buffer, params_shape.clone(), device.clone()),
            Tensor::from_buffer(m_out_buffer, params_shape.clone(), device.clone()),
            Tensor::from_buffer(v_out_buffer, params_shape, device.clone()),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// NAdam optimizer step (Nesterov-accelerated Adam)
    ///
    /// **Deep Debt**: Faster convergence than vanilla Adam
    ///
    /// # Arguments
    /// - `gradients`: Gradients tensor
    /// - `learning_rate`: Learning rate (typically 0.001)
    /// - `beta1`: First moment decay (typically 0.9)
    /// - `beta2`: Second moment decay (typically 0.999)
    /// - `step`: Current training step
    /// - `m`: First moment state (None for initialization)
    /// - `v`: Second moment state (None for initialization)
    ///
    /// # Returns
    /// - (updated_params, updated_m, updated_v)
    ///
    /// # Example
    /// ```rust,ignore
    /// let (params, m, v) = params.nadam_step(
    ///     &grads, 0.001, 0.9, 0.999, step, 
    ///     Some(&m), Some(&v)
    /// )?;
    /// ```
    pub fn nadam_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        m: Option<&Self>,
        v: Option<&Self>,
    ) -> Result<(Self, Self, Self)> {
        NAdam::new(
            self,
            gradients.clone(),
            learning_rate,
            beta1,
            beta2,
            step,
            m.cloned(),
            v.cloned(),
        )?
        .execute()
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device;

    #[tokio::test]
    async fn test_nadam_gpu_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0; 100], vec![100], device.clone())
            .await
            .unwrap();
        let grads = Tensor::from_vec_on(vec![0.01; 100], vec![100], device)
            .await
            .unwrap();

        let (new_params, m, v) = params.nadam_step(&grads, 0.001, 0.9, 0.999, 1, None, None).unwrap();

        assert_eq!(new_params.shape(), &[100]);
        assert_eq!(m.shape(), &[100]);
        assert_eq!(v.shape(), &[100]);

        let data = new_params.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
        // Parameters should decrease with positive gradients
        assert!(data[0] < 1.0);
    }

    #[tokio::test]
    async fn test_nadam_gpu_with_state() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let grads = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
            .await
            .unwrap();

        // First step
        let (params1, m1, v1) = params.nadam_step(&grads, 0.01, 0.9, 0.999, 1, None, None).unwrap();

        // Second step with state
        let (params2, _m2, _v2) = params1.nadam_step(&grads, 0.01, 0.9, 0.999, 2, Some(&m1), Some(&v1)).unwrap();

        let data = params2.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_nadam_gpu_convergence() {
        let device = crate::device::test_pool::get_test_device().await;
        let mut params = Tensor::from_vec_on(vec![10.0; 50], vec![50], device.clone())
            .await
            .unwrap();
        let grads = Tensor::from_vec_on(vec![0.1; 50], vec![50], device)
            .await
            .unwrap();

        let mut m = None;
        let mut v = None;

        // Multiple steps should decrease parameters
        for step in 1..=5 {
            let (new_params, new_m, new_v) = params.nadam_step(&grads, 0.01, 0.9, 0.999, step, m.as_ref(), v.as_ref()).unwrap();
            params = new_params;
            m = Some(new_m);
            v = Some(new_v);
        }

        let final_data = params.to_vec().unwrap();
        assert!(final_data[0] < 10.0); // Should have decreased
    }
}
