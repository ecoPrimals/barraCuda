// SPDX-License-Identifier: AGPL-3.0-or-later
//! PPPM compute with CPU FFT (k-space)

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::{BarracudaError, Result};
use crate::linalg::sparse::SparseBuffers;
use wgpu::util::DeviceExt;

use super::super::pppm_buffers::PppmCpuFft;
use super::super::short_range::dipole_correction;
use super::PppmGpu;

/// Wrap positions to [0, L) for periodic boundary conditions (matches CPU)
fn wrap_positions(positions: &[f64], box_dims: [f64; 3]) -> Vec<f64> {
    positions
        .chunks_exact(3)
        .flat_map(|c| {
            [
                (c[0] / box_dims[0]).floor().mul_add(-box_dims[0], c[0]),
                (c[1] / box_dims[1]).floor().mul_add(-box_dims[1], c[1]),
                (c[2] / box_dims[2]).floor().mul_add(-box_dims[2], c[2]),
            ]
        })
        .collect()
}

/// Full PPPM with CPU FFT
pub async fn compute_with_kspace(
    pppm: &PppmGpu,
    positions: &[f64],
    charges: &[f64],
) -> Result<(Vec<f64>, f64)> {
    let n = charges.len();
    if positions.len() != n * 3 {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "positions length {} != charges length {} * 3",
                positions.len(),
                n
            ),
        });
    }
    // Wrap positions to box (matches CPU charge_spread/force_interpolation)
    let positions = wrap_positions(positions, pppm.params().box_dims);
    let order = pppm.params().interpolation_order;
    let [kx, ky, kz] = pppm.params().mesh_dims;
    let mesh_size = kx * ky * kz;
    let o3 = order * order * order;

    let (device, queue, wgpu_device, layouts, pipelines) = (
        pppm.device(),
        pppm.queue(),
        pppm.wgpu_device(),
        pppm.layouts(),
        pppm.pipelines(),
    );

    let positions_buffer = SparseBuffers::f64_from_slice_raw(device, "positions", &positions);
    let charges_buffer = SparseBuffers::f64_from_slice_raw(device, "charges", charges);
    let coeffs_size = n * order * 3;
    let coeffs_buffer = SparseBuffers::f64_zeros_raw(device, "coeffs", coeffs_size);
    let derivs_buffer = SparseBuffers::f64_zeros_raw(device, "derivs", coeffs_size);
    let base_idx_buffer = SparseBuffers::i32_zeros_raw(device, "base_idx", n * 3);
    let per_particle_mesh_buffer =
        SparseBuffers::f64_zeros_raw(device, "per_particle_mesh", n * o3);
    let bspline_params: Vec<f64> = vec![
        n as f64,
        order as f64,
        kx as f64,
        ky as f64,
        kz as f64,
        pppm.params().box_dims[0],
        pppm.params().box_dims[1],
        pppm.params().box_dims[2],
    ];
    let bspline_params_buffer =
        SparseBuffers::f64_from_slice_raw(device, "bspline_params", &bspline_params);
    let spread_params: Vec<f64> = vec![n as f64, order as f64, kx as f64, ky as f64, kz as f64];
    let spread_params_buffer =
        SparseBuffers::f64_from_slice_raw(device, "spread_params", &spread_params);

    let bspline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bspline_bg"),
        layout: &layouts.bspline,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: positions_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coeffs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: derivs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: base_idx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bspline_params_buffer.as_entire_binding(),
            },
        ],
    });
    let charge_spread_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spread_bg"),
        layout: &layouts.charge_spread,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: charges_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coeffs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: base_idx_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: per_particle_mesh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: spread_params_buffer.as_entire_binding(),
            },
        ],
    });
    let particle_workgroups = (n as u32).div_ceil(WORKGROUP_SIZE_COMPACT);

    let mut encoder = wgpu_device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("PPPM Phase 1"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("B-spline"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.bspline);
        pass.set_bind_group(0, Some(&bspline_bind_group), &[]);
        pass.dispatch_workgroups(particle_workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Charge Spread"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.charge_spread);
        pass.set_bind_group(0, Some(&charge_spread_bind_group), &[]);
        pass.dispatch_workgroups(particle_workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let coeffs = SparseBuffers::read_f64_raw(wgpu_device.as_ref(), &coeffs_buffer, coeffs_size)?;
    let derivs = SparseBuffers::read_f64_raw(wgpu_device.as_ref(), &derivs_buffer, coeffs_size)?;
    let base_idx = SparseBuffers::read_i32_raw(wgpu_device.as_ref(), &base_idx_buffer, n * 3)?;
    let per_particle_mesh =
        SparseBuffers::read_f64_raw(wgpu_device.as_ref(), &per_particle_mesh_buffer, n * o3)?;

    let mut charge_mesh = vec![0.0f64; mesh_size];
    for i in 0..n {
        let bx = base_idx[i * 3];
        let by = base_idx[i * 3 + 1];
        let bz = base_idx[i * 3 + 2];
        let mut local_idx = 0;
        for jx in 0..order {
            let ix = ((bx + jx as i32) % kx as i32 + kx as i32) as usize % kx;
            for jy in 0..order {
                let iy = ((by + jy as i32) % ky as i32 + ky as i32) as usize % ky;
                for jz in 0..order {
                    let iz = ((bz + jz as i32) % kz as i32 + kz as i32) as usize % kz;
                    let mesh_idx = ix * ky * kz + iy * kz + iz;
                    charge_mesh[mesh_idx] += per_particle_mesh[i * o3 + local_idx];
                    local_idx += 1;
                }
            }
        }
    }

    let rho_k = PppmCpuFft::forward_3d(&charge_mesh, kx, ky, kz);
    let phi_k = pppm.greens().apply(&rho_k);
    let volume = pppm.params().box_dims[0] * pppm.params().box_dims[1] * pppm.params().box_dims[2];
    let e_kspace = pppm.greens().kspace_energy(&rho_k, volume);
    let potential_values = PppmCpuFft::inverse_3d(&phi_k, kx, ky, kz);

    let potential_buffer =
        SparseBuffers::f64_from_slice_raw(device, "potential", &potential_values);
    let forces_buffer = SparseBuffers::f64_zeros_raw(device, "forces", n * 3);
    let pe_buffer = SparseBuffers::f64_zeros_raw(device, "pe", n);
    let interp_params: Vec<f64> = vec![
        n as f64,
        order as f64,
        kx as f64,
        ky as f64,
        kz as f64,
        pppm.params().box_dims[0],
        pppm.params().box_dims[1],
        pppm.params().box_dims[2],
    ];
    let interp_params_buffer =
        SparseBuffers::f64_from_slice_raw(device, "interp_params", &interp_params);
    let coeffs_buffer2 = SparseBuffers::f64_from_slice_raw(device, "coeffs2", &coeffs);
    let derivs_buffer2 = SparseBuffers::f64_from_slice_raw(device, "derivs2", &derivs);
    let base_idx_bytes: &[u8] = bytemuck::cast_slice(&base_idx);
    let base_idx_buffer2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("base_idx2"),
        contents: base_idx_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });

    let force_interp_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("force_interp_bg"),
        layout: &layouts.force_interp,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: charges_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: coeffs_buffer2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: derivs_buffer2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: base_idx_buffer2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: potential_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: forces_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: interp_params_buffer.as_entire_binding(),
            },
        ],
    });
    let erfc_params: Vec<f64> = vec![
        n as f64,
        pppm.params().alpha,
        pppm.params().real_cutoff * pppm.params().real_cutoff,
        pppm.params().box_dims[0],
        pppm.params().box_dims[1],
        pppm.params().box_dims[2],
        pppm.params().coulomb_constant,
    ];
    let erfc_params_buffer = SparseBuffers::f64_from_slice_raw(device, "erfc_params", &erfc_params);
    let erfc_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("erfc_bg"),
        layout: &layouts.erfc_forces,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: positions_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: charges_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: forces_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: pe_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: erfc_params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = wgpu_device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("PPPM Phase 3"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Force Interp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.force_interp);
        pass.set_bind_group(0, Some(&force_interp_bind_group), &[]);
        pass.dispatch_workgroups(particle_workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("erfc Forces"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.erfc_forces);
        pass.set_bind_group(0, Some(&erfc_bind_group), &[]);
        pass.dispatch_workgroups(particle_workgroups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Self Energy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.self_energy);
        pass.set_bind_group(0, Some(&erfc_bind_group), &[]);
        pass.dispatch_workgroups(particle_workgroups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let forces = SparseBuffers::read_f64_raw(wgpu_device.as_ref(), &forces_buffer, n * 3)?;
    let pe_values = SparseBuffers::read_f64_raw(wgpu_device.as_ref(), &pe_buffer, n)?;
    let pos_arrays: Vec<[f64; 3]> = positions
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let e_dipole = dipole_correction(
        &pos_arrays,
        charges,
        pppm.params().box_dims,
        pppm.params().coulomb_constant,
    );
    let e_short_and_self: f64 = pe_values.iter().sum();
    let total_energy = e_kspace + e_short_and_self + e_dipole;
    Ok((forces, total_energy))
}
