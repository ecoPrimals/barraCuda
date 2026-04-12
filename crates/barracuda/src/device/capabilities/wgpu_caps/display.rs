// SPDX-License-Identifier: AGPL-3.0-or-later
//! `Display` impl for [`DeviceCapabilities`] — human-readable summary.

use super::{BYTES_PER_MB, DeviceCapabilities, WorkloadType};
use std::fmt;

impl fmt::Display for DeviceCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Device Capabilities:")?;
        writeln!(f, "  Name: {}", self.device_name)?;
        writeln!(f, "  Type: {:?}", self.device_type)?;
        writeln!(
            f,
            "  Vendor: {} (0x{:04X})",
            self.vendor_name(),
            self.vendor
        )?;
        writeln!(f, "  Backend: {:?}", self.backend)?;
        writeln!(f)?;
        writeln!(f, "Memory:")?;
        writeln!(
            f,
            "  Max Buffer Size: {} MB",
            self.max_buffer_size / BYTES_PER_MB
        )?;
        writeln!(
            f,
            "  Max Allocation: {} MB",
            self.max_allocation_size() / BYTES_PER_MB
        )?;
        writeln!(f)?;
        writeln!(f, "Compute:")?;
        writeln!(f, "  Max Workgroup Size: {:?}", self.max_workgroup_size)?;
        writeln!(
            f,
            "  Max Invocations/Workgroup: {}",
            self.max_compute_invocations_per_workgroup
        )?;
        writeln!(
            f,
            "  Max Compute Workgroups: {:?}",
            self.max_compute_workgroups
        )?;
        writeln!(f)?;
        writeln!(f, "Optimal Configurations:")?;
        writeln!(
            f,
            "  Element-wise: {} threads",
            self.optimal_workgroup_size(WorkloadType::ElementWise)
        )?;
        writeln!(
            f,
            "  MatMul: {} threads (tile: {})",
            self.optimal_workgroup_size(WorkloadType::MatMul),
            self.optimal_matmul_tile_size()
        )?;
        writeln!(
            f,
            "  Reduction: {} threads",
            self.optimal_workgroup_size(WorkloadType::Reduction)
        )?;
        writeln!(
            f,
            "  FHE: {} threads",
            self.optimal_workgroup_size(WorkloadType::FHE)
        )?;
        writeln!(
            f,
            "  Convolution: {:?}",
            self.optimal_workgroup_size_2d(WorkloadType::Convolution)
        )?;
        writeln!(f)?;
        if self.has_subgroup_info() {
            writeln!(
                f,
                "Subgroup: {}-{} lanes",
                self.subgroup_min_size, self.subgroup_max_size,
            )?;
        } else {
            writeln!(f, "Subgroup: not reported")?;
        }
        writeln!(f)?;
        writeln!(f, "Features:")?;
        writeln!(
            f,
            "  f64 shaders: {}",
            if self.f64_shaders { "Yes" } else { "No" }
        )?;
        writeln!(
            f,
            "  f64 shared memory: {}",
            if self.f64_shared_memory {
                "Yes"
            } else {
                "No (naga/SPIR-V bug — use DF64)"
            }
        )?;
        writeln!(
            f,
            "  FHE Support: {}",
            if self.supports_fhe() { "Yes" } else { "No" }
        )?;
        writeln!(
            f,
            "  High Performance: {}",
            if self.is_high_performance() {
                "Yes"
            } else {
                "No"
            }
        )?;
        Ok(())
    }
}
