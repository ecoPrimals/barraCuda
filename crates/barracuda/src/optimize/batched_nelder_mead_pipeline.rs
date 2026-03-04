// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pipeline and BGL helpers for batched Nelder-Mead GPU.

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::uniform_bgl_entry;
use crate::device::WgpuDevice;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BatchedSimplexParams {
    pub n_problems: u32,
    pub n: u32,
    pub n_points: u32,
    pub _pad: u32,
    pub alpha: f64,
    pub gamma: f64,
    pub rho: f64,
    pub sigma: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BatchedContractParams {
    pub n_problems: u32,
    pub n: u32,
    pub rho: f64,
    pub _pad: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct BatchedShrinkParams {
    pub n_problems: u32,
    pub n: u32,
    pub n_points: u32,
    pub _pad: u32,
    pub sigma: f64,
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn create_centroid_bgl(device: &WgpuDevice) -> wgpu::BindGroupLayout {
    device
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nm_centroid_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                bgl_entry(1, true),
                bgl_entry(2, true),
                bgl_entry(3, true),
                bgl_entry(4, false),
                bgl_entry(5, false),
            ],
        })
}

pub fn create_contract_bgl(device: &WgpuDevice) -> wgpu::BindGroupLayout {
    device
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nm_contract_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                bgl_entry(1, true),
                bgl_entry(2, true),
                bgl_entry(3, true),
                bgl_entry(4, true),
                bgl_entry(5, true),
                bgl_entry(6, false),
            ],
        })
}

pub fn create_shrink_bgl(device: &WgpuDevice) -> wgpu::BindGroupLayout {
    device
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nm_shrink_bgl"),
            entries: &[
                uniform_bgl_entry(0),
                bgl_entry(1, true),
                bgl_entry(2, false),
            ],
        })
}

pub fn create_f64_buffer(device: &WgpuDevice, label: &str, data: &[f64]) -> wgpu::Buffer {
    device
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

pub fn create_storage_buffer(
    device: &WgpuDevice,
    label: &str,
    size: u64,
    copy_src: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

pub fn run_centroid_reflect(
    device: &WgpuDevice,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    params_buf: &wgpu::Buffer,
    simplex_buf: &wgpu::Buffer,
    f_vals_buf: &wgpu::Buffer,
    worst_idx_buf: &wgpu::Buffer,
    centroid_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    n_problems: usize,
    n: usize,
    entry: &str,
) {
    let pl = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nm_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nm_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: simplex_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: f_vals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: worst_idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: centroid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pl),
            module: shader,
            entry_point: Some(entry),
            cache: device.pipeline_cache(),
            compilation_options: Default::default(),
        });
    let mut enc = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("nm_enc"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(
            (n_problems * n).div_ceil(WORKGROUP_SIZE_1D as usize) as u32,
            1,
            1,
        );
    }
    device.submit_and_poll_inner(Some(enc.finish()));
}

pub fn run_contract(
    device: &WgpuDevice,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    params_buf: &wgpu::Buffer,
    simplex_buf: &wgpu::Buffer,
    worst_idx_buf: &wgpu::Buffer,
    centroid_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    inside_buf: &wgpu::Buffer,
    contract_out_buf: &wgpu::Buffer,
    n_problems: usize,
    n: usize,
) {
    let pl = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nm_contract_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nm_contract_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: simplex_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: worst_idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: centroid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: inside_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: contract_out_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batched_contract"),
            layout: Some(&pl),
            module: shader,
            entry_point: Some("batched_contract"),
            cache: device.pipeline_cache(),
            compilation_options: Default::default(),
        });
    let mut enc = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("nm_contract_enc"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(
            (n_problems * n).div_ceil(WORKGROUP_SIZE_1D as usize) as u32,
            1,
            1,
        );
    }
    device.submit_and_poll_inner(Some(enc.finish()));
}

pub fn apply_nm_step(
    simplex: &mut [f64],
    f_vals: &mut [f64],
    reflect_pts: &[f64],
    f_reflect: &[f64],
    centroid: &[f64],
    best_idx: &[u32],
    worst_idx: &[u32],
    need_contract: &mut [bool],
    inside_contract: &mut [u32],
    active: &[usize],
    n: usize,
    n_points: usize,
    gamma: f64,
    f_values: &mut impl FnMut(&[f64]) -> Vec<f64>,
    n_problems: usize,
) {
    for &p in active {
        let base = p * n_points;
        let best_idx_p = best_idx[p] as usize;
        let worst_idx_p = worst_idx[p] as usize;
        let second_worst: usize = (0..n_points)
            .filter(|&i| i != worst_idx_p)
            .max_by(|&a, &b| {
                f_vals[base + a]
                    .partial_cmp(&f_vals[base + b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_default();
        let f_r = f_reflect[p];
        let f_best = f_vals[base + best_idx_p];
        let f_sw = f_vals[base + second_worst];
        let f_worst = f_vals[base + worst_idx_p];

        if f_r < f_best {
            let mut expand = vec![0.0; n_problems * n];
            for j in 0..n {
                expand[p * n + j] =
                    centroid[p * n + j] + gamma * (reflect_pts[p * n + j] - centroid[p * n + j]);
            }
            let f_expand = f_values(&expand);
            if f_expand[p] < f_r {
                for j in 0..n {
                    simplex[p * n_points * n + worst_idx_p * n + j] = expand[p * n + j];
                }
                f_vals[base + worst_idx_p] = f_expand[p];
            } else {
                for j in 0..n {
                    simplex[p * n_points * n + worst_idx_p * n + j] = reflect_pts[p * n + j];
                }
                f_vals[base + worst_idx_p] = f_r;
            }
        } else if f_r < f_sw {
            for j in 0..n {
                simplex[p * n_points * n + worst_idx_p * n + j] = reflect_pts[p * n + j];
            }
            f_vals[base + worst_idx_p] = f_r;
        } else {
            need_contract[p] = true;
            inside_contract[p] = if f_r < f_worst { 0 } else { 1 };
        }
    }
}

pub fn run_shrink(
    device: &WgpuDevice,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    params_buf: &wgpu::Buffer,
    best_idx_buf: &wgpu::Buffer,
    simplex_buf: &wgpu::Buffer,
    n_problems: usize,
    n: usize,
    n_points: usize,
) {
    let pl = device
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nm_shrink_pl"),
            bind_group_layouts: &[bgl],
            immediate_size: 0,
        });
    let bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nm_shrink_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: best_idx_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: simplex_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline = device
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("batched_shrink"),
            layout: Some(&pl),
            module: shader,
            entry_point: Some("batched_shrink"),
            cache: device.pipeline_cache(),
            compilation_options: Default::default(),
        });
    let mut enc = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("nm_shrink_enc"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(
            (n_problems * n_points * n).div_ceil(WORKGROUP_SIZE_1D as usize) as u32,
            1,
            1,
        );
    }
    device.submit_and_poll_inner(Some(enc.finish()));
}
