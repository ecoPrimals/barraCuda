// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU interpreter for naga IR compute shaders.

use std::collections::BTreeMap;

use naga::{AddressSpace, Module};

use crate::error::{NagaExecError, Result};
use crate::invocation::{DispatchCoords, InvocationContext};
use crate::sim_buffer::SimBuffer;
use crate::workgroup::{InvocationState, WorkgroupMemory, split_at_barriers};

/// CPU execution context for a validated WGSL compute shader.
///
/// Parse and validate once, dispatch many times with different buffers.
pub struct NagaExecutor {
    module: Module,
    info: naga::valid::ModuleInfo,
    entry_point_index: usize,
}

impl NagaExecutor {
    /// Parse and validate WGSL, ready for CPU execution.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::Parse`] if the WGSL source is invalid,
    /// [`NagaExecError::Validation`] if naga validation fails, or
    /// [`NagaExecError::EntryPointNotFound`] if the entry point doesn't exist.
    pub fn new(wgsl_source: &str, entry_point: &str) -> Result<Self> {
        let module = naga::front::wgsl::parse_str(wgsl_source)
            .map_err(|e| NagaExecError::Parse(format!("{e}")))?;

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .map_err(|e| NagaExecError::Validation(format!("{e}")))?;

        let entry_point_index = module
            .entry_points
            .iter()
            .position(|ep| ep.name == entry_point && ep.stage == naga::ShaderStage::Compute)
            .ok_or_else(|| NagaExecError::EntryPointNotFound(entry_point.to_string()))?;

        Ok(Self {
            module,
            info,
            entry_point_index,
        })
    }

    /// Execute the compute shader on CPU, simulating `dispatch(x, y, z)`.
    ///
    /// Bindings map `(group, binding)` -> `SimBuffer`. After dispatch,
    /// read-write storage buffers contain the shader's output.
    ///
    /// Supports `var<workgroup>` shared memory, `workgroupBarrier()`, and
    /// atomic operations. Barriers are handled by executing all invocations
    /// in a workgroup together, splitting execution at barrier points.
    ///
    /// # Errors
    ///
    /// Returns errors for missing bindings, unsupported IR nodes, etc.
    pub fn dispatch(
        &self,
        workgroups: (u32, u32, u32),
        bindings: &mut BTreeMap<(u32, u32), SimBuffer>,
    ) -> Result<()> {
        let ep = &self.module.entry_points[self.entry_point_index];
        let wg_size = ep.workgroup_size;
        let has_workgroup_vars = self.has_workgroup_vars();

        let num_wg = [workgroups.0, workgroups.1, workgroups.2];
        for wg_z in 0..workgroups.2 {
            for wg_y in 0..workgroups.1 {
                for wg_x in 0..workgroups.0 {
                    if has_workgroup_vars {
                        self.dispatch_workgroup_barrier_aware(
                            [wg_x, wg_y, wg_z],
                            wg_size,
                            num_wg,
                            bindings,
                        )?;
                    } else {
                        self.dispatch_workgroup_simple(
                            [wg_x, wg_y, wg_z],
                            wg_size,
                            num_wg,
                            bindings,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn has_workgroup_vars(&self) -> bool {
        self.module
            .global_variables
            .iter()
            .any(|(_, gv)| gv.space == AddressSpace::WorkGroup)
    }

    fn dispatch_workgroup_simple(
        &self,
        wg_id: [u32; 3],
        wg_size: [u32; 3],
        num_workgroups: [u32; 3],
        bindings: &mut BTreeMap<(u32, u32), SimBuffer>,
    ) -> Result<()> {
        let ep = &self.module.entry_points[self.entry_point_index];
        for local_z in 0..wg_size[2] {
            for local_y in 0..wg_size[1] {
                for local_x in 0..wg_size[0] {
                    let global_id = [
                        wg_id[0] * wg_size[0] + local_x,
                        wg_id[1] * wg_size[1] + local_y,
                        wg_id[2] * wg_size[2] + local_z,
                    ];
                    let local_id = [local_x, local_y, local_z];
                    let mut shared = WorkgroupMemory::default();
                    let mut ctx = InvocationContext::new(
                        &self.module,
                        &self.info,
                        &ep.function,
                        bindings,
                        &mut shared,
                        DispatchCoords {
                            global_id,
                            local_id,
                            workgroup_id: wg_id,
                            workgroup_size: wg_size,
                            num_workgroups,
                        },
                    );
                    ctx.execute_body(&ep.function.body)?;
                }
            }
        }
        Ok(())
    }

    /// Barrier-aware workgroup dispatch: all invocations execute the body
    /// together, splitting at `workgroupBarrier()` points.
    ///
    /// At each barrier, all invocations have completed their stores to
    /// shared memory, so subsequent reads see consistent data.
    fn dispatch_workgroup_barrier_aware(
        &self,
        wg_id: [u32; 3],
        wg_size: [u32; 3],
        num_workgroups: [u32; 3],
        bindings: &mut BTreeMap<(u32, u32), SimBuffer>,
    ) -> Result<()> {
        let ep = &self.module.entry_points[self.entry_point_index];
        let total = u64::from(wg_size[0])
            .checked_mul(u64::from(wg_size[1]))
            .and_then(|n| n.checked_mul(u64::from(wg_size[2])))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                NagaExecError::Overflow("workgroup size product overflows usize".into())
            })?;

        let mut invocations: Vec<([u32; 3], [u32; 3])> = Vec::with_capacity(total);
        for local_z in 0..wg_size[2] {
            for local_y in 0..wg_size[1] {
                for local_x in 0..wg_size[0] {
                    let global_id = [
                        wg_id[0] * wg_size[0] + local_x,
                        wg_id[1] * wg_size[1] + local_y,
                        wg_id[2] * wg_size[2] + local_z,
                    ];
                    invocations.push((global_id, [local_x, local_y, local_z]));
                }
            }
        }

        let segments = split_at_barriers(&ep.function.body);
        let mut shared = WorkgroupMemory::new(&self.module);
        let mut inv_states: Vec<InvocationState> =
            invocations.iter().map(|_| InvocationState::new()).collect();

        for segment in &segments {
            for (i, (global_id, local_id)) in invocations.iter().enumerate() {
                let mut ctx = InvocationContext::new(
                    &self.module,
                    &self.info,
                    &ep.function,
                    bindings,
                    &mut shared,
                    DispatchCoords {
                        global_id: *global_id,
                        local_id: *local_id,
                        workgroup_id: wg_id,
                        workgroup_size: wg_size,
                        num_workgroups,
                    },
                );
                ctx.restore_state(&inv_states[i]);
                ctx.execute_body(segment)?;
                inv_states[i] = ctx.save_state();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
