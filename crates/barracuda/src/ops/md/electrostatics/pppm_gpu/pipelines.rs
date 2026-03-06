// SPDX-License-Identifier: AGPL-3.0-or-later
//! PPPM bind group layouts and pipeline creation
//!
//! Extracted from `pppm_layouts` for `pppm_gpu` modularity.
//! Contains all bind group layout definitions and compute pipeline creation.

/// PPPM bind group layout helpers for creating individual bind group layouts.
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

    /// B-spline charge assignment bind group layout.
    #[must_use]
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

    /// Charge spreading to grid bind group layout.
    #[must_use]
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

    /// Greens function application bind group layout.
    #[must_use]
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

    /// Force interpolation from grid bind group layout.
    #[must_use]
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

    /// Real-space erfc forces bind group layout.
    #[must_use]
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

/// PPPM bind group layout collection for all pipeline stages.
pub struct PppmBindGroupLayouts {
    /// B-spline charge assignment layout
    pub bspline: wgpu::BindGroupLayout,
    /// Charge spreading layout
    pub charge_spread: wgpu::BindGroupLayout,
    /// Greens function application layout
    pub greens_apply: wgpu::BindGroupLayout,
    /// Force interpolation layout
    pub force_interp: wgpu::BindGroupLayout,
    /// Real-space erfc forces layout
    pub erfc_forces: wgpu::BindGroupLayout,
}

impl PppmBindGroupLayouts {
    /// Create all PPPM bind group layouts for a device.
    #[must_use]
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

/// PPPM compute pipeline collection for all electrostatics stages.
pub struct PppmPipelines {
    /// B-spline charge assignment pipeline
    pub bspline: wgpu::ComputePipeline,
    /// Charge spreading pipeline
    pub charge_spread: wgpu::ComputePipeline,
    /// Greens function application pipeline
    pub greens_apply: wgpu::ComputePipeline,
    /// Force interpolation pipeline
    pub force_interp: wgpu::ComputePipeline,
    /// Real-space erfc forces pipeline
    pub erfc_forces: wgpu::ComputePipeline,
    /// Self-energy correction pipeline
    pub self_energy: wgpu::ComputePipeline,
}

impl PppmPipelines {
    /// Create all PPPM compute pipelines from shader modules.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
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
                    immediate_size: 0,
                }),
            ),
            module: bspline_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let charge_spread = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_charge_spread_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_charge_spread_layout"),
                    bind_group_layouts: &[&layouts.charge_spread],
                    immediate_size: 0,
                }),
            ),
            module: charge_spread_module,
            entry_point: Some("spread_per_particle"),
            cache: None,
            compilation_options: Default::default(),
        });

        let greens_apply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_greens_apply_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_greens_apply_layout"),
                    bind_group_layouts: &[&layouts.greens_apply],
                    immediate_size: 0,
                }),
            ),
            module: greens_apply_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let force_interp = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_force_interp_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_force_interp_layout"),
                    bind_group_layouts: &[&layouts.force_interp],
                    immediate_size: 0,
                }),
            ),
            module: force_interp_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let erfc_forces = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_erfc_forces_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_erfc_forces_layout"),
                    bind_group_layouts: &[&layouts.erfc_forces],
                    immediate_size: 0,
                }),
            ),
            module: erfc_forces_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });

        let self_energy = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pppm_self_energy_pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pppm_self_energy_layout"),
                    bind_group_layouts: &[&layouts.erfc_forces],
                    immediate_size: 0,
                }),
            ),
            module: erfc_forces_module,
            entry_point: Some("self_energy"),
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
