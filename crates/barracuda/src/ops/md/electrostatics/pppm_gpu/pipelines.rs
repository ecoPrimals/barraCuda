//! PPPM bind group layouts and pipeline creation
//!
//! Extracted from pppm_layouts for pppm_gpu modularity.
//! Contains all bind group layout definitions and compute pipeline creation.

use std::sync::Arc;

/// PPPM bind group layout helpers (for creating individual layouts)
pub struct PppmLayouts;

impl PppmLayouts {
    fn storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn bspline(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pppm_bspline_bgl"),
            entries: &[
                Self::storage_ro(0),
                Self::storage_rw(1),
                Self::storage_rw(2),
                Self::storage_rw(3),
                Self::storage_ro(4),
            ],
        })
    }

    pub fn charge_spread(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pppm_charge_spread_bgl"),
            entries: &[
                Self::storage_ro(0),
                Self::storage_ro(1),
                Self::storage_ro(2),
                Self::storage_rw(3),
                Self::storage_ro(4),
            ],
        })
    }

    pub fn greens_apply(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pppm_greens_apply_bgl"),
            entries: &[
                Self::storage_ro(0),
                Self::storage_ro(1),
                Self::storage_rw(2),
                Self::storage_rw(3),
                Self::storage_ro(4),
                Self::storage_ro(5),
            ],
        })
    }

    pub fn force_interp(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pppm_force_interp_bgl"),
            entries: &[
                Self::storage_ro(0),
                Self::storage_ro(1),
                Self::storage_ro(2),
                Self::storage_ro(3),
                Self::storage_ro(4),
                Self::storage_rw(5),
                Self::storage_ro(6),
            ],
        })
    }

    pub fn erfc_forces(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pppm_erfc_forces_bgl"),
            entries: &[
                Self::storage_ro(0),
                Self::storage_ro(1),
                Self::storage_rw(2),
                Self::storage_rw(3),
                Self::storage_ro(4),
            ],
        })
    }
}

/// PPPM bind group layout collection
pub struct PppmBindGroupLayouts {
    pub bspline: wgpu::BindGroupLayout,
    pub charge_spread: wgpu::BindGroupLayout,
    pub greens_apply: wgpu::BindGroupLayout,
    pub force_interp: wgpu::BindGroupLayout,
    pub erfc_forces: wgpu::BindGroupLayout,
}

impl PppmBindGroupLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            bspline: PppmLayouts::bspline(device),
            charge_spread: PppmLayouts::charge_spread(device),
            greens_apply: PppmLayouts::greens_apply(device),
            force_interp: PppmLayouts::force_interp(device),
            erfc_forces: PppmLayouts::erfc_forces(device),
        }
    }
}

/// PPPM compute pipeline collection
pub struct PppmPipelines {
    pub bspline: wgpu::ComputePipeline,
    pub charge_spread: wgpu::ComputePipeline,
    pub greens_apply: wgpu::ComputePipeline,
    pub force_interp: wgpu::ComputePipeline,
    pub erfc_forces: wgpu::ComputePipeline,
    pub self_energy: wgpu::ComputePipeline,
}

impl PppmPipelines {
    pub fn new(
        device: &Arc<wgpu::Device>,
        layouts: &PppmBindGroupLayouts,
        bspline_module: &wgpu::ShaderModule,
        charge_spread_module: &wgpu::ShaderModule,
        greens_apply_module: &wgpu::ShaderModule,
        force_interp_module: &wgpu::ShaderModule,
        erfc_forces_module: &wgpu::ShaderModule,
    ) -> Self {
        let bspline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_bspline_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_bspline_layout"),
                    bind_group_layouts: &[&layouts.bspline],
                    push_constant_ranges: &[],
                }),
            ),
            module: bspline_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let charge_spread = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_charge_spread_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_charge_spread_layout"),
                    bind_group_layouts: &[&layouts.charge_spread],
                    push_constant_ranges: &[],
                }),
            ),
            module: charge_spread_module,
            entry_point: "spread_per_particle",
            cache: None,
            compilation_options: Default::default(),
        });

        let greens_apply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_greens_apply_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_greens_apply_layout"),
                    bind_group_layouts: &[&layouts.greens_apply],
                    push_constant_ranges: &[],
                }),
            ),
            module: greens_apply_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let force_interp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_force_interp_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_force_interp_layout"),
                    bind_group_layouts: &[&layouts.force_interp],
                    push_constant_ranges: &[],
                }),
            ),
            module: force_interp_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let erfc_forces = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_erfc_forces_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_erfc_forces_layout"),
                    bind_group_layouts: &[&layouts.erfc_forces],
                    push_constant_ranges: &[],
                }),
            ),
            module: erfc_forces_module,
            entry_point: "main",
            cache: None,
            compilation_options: Default::default(),
        });

        let self_energy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_self_energy_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_self_energy_layout"),
                    bind_group_layouts: &[&layouts.erfc_forces],
                    push_constant_ranges: &[],
                }),
            ),
            module: erfc_forces_module,
            entry_point: "self_energy",
            cache: None,
            compilation_options: Default::default(),
        });

        Self {
            bspline,
            charge_spread,
            greens_apply,
            force_interp,
            erfc_forces,
            self_energy,
        }
    }
}
