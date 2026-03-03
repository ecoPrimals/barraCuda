// SPDX-License-Identifier: AGPL-3.0-only

//! Hill Gate — Two-input Hill AND gate (f64 pipeline).
//!
//! Computes f(a, b) = vmax × H(a, K_a, n_a) × H(b, K_b, n_b) where
//! H(x, K, n) = x^n / (K^n + x^n) is the Hill function. Used for regulatory
//! network signal integration.
//!
//! Provenance: neuralSpring metalForge → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;

pub const WGSL_HILL_GATE: &str = include_str!("../../shaders/bio/hill_gate.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_HILL_GATE_F64: &str = include_str!("../../shaders/bio/hill_gate_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HillGateParams {
    pub n_a: u32,
    pub n_b: u32,
    pub mode: u32, // 0 = paired, 1 = grid
    pub _pad: u32,
    pub k_a: f64,
    pub k_b: f64,
    pub n_a_exp: f64,
    pub n_b_exp: f64,
    pub vmax: f64,
    pub _pad2: f64,
}

/// Hill gate GPU kernel (f64 pipeline).
pub struct HillGateGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl HillGateGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HillGate BGL"),
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

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HillGate Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let module = device.compile_shader_f64(WGSL_HILL_GATE_F64, Some("HillGate f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HillGate Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Compute Hill gate. Mode 0: paired (output[i] = f(a[i], b[i])).
    /// Mode 1: grid (output[ix*n_b + iy] = f(a[ix], b[iy])).
    pub fn dispatch(
        &self,
        input_a: &wgpu::Buffer,
        input_b: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &HillGateParams,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HillGate Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let workgroups = if params.mode == 0 {
            params.n_a.div_ceil(256)
        } else {
            (params.n_a * params.n_b).div_ceil(256)
        };

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HillGate BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = d.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("HillGate Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HillGate Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::{HillGateGpu, HillGateParams, WGSL_HILL_GATE, WGSL_HILL_GATE_F64};

    #[test]
    fn sanity_constants_exported() {
        assert!(!WGSL_HILL_GATE.is_empty());
        assert!(WGSL_HILL_GATE.contains("fn main"));
        assert!(WGSL_HILL_GATE.contains("HillGateParams"));
        assert!(std::any::type_name::<HillGateGpu>().contains("HillGateGpu"));
        assert!(std::any::type_name::<HillGateParams>().contains("HillGateParams"));
    }

    #[test]
    fn f64_shader_contains_main() {
        assert!(WGSL_HILL_GATE_F64.contains("fn main"));
        assert!(WGSL_HILL_GATE_F64.contains("f64"));
    }

    #[test]
    fn f64_shader_compiles_via_naga() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        device.compile_shader_f64(WGSL_HILL_GATE_F64, Some("hill_gate_f64"));
    }
}
