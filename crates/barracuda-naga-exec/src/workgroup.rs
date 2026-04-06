// SPDX-License-Identifier: AGPL-3.0-or-later
//! Workgroup-scoped shared memory and barrier-aware execution support.
//!
//! Extracted from [`crate::executor`] to keep per-file line count under 1000
//! while preserving the logical grouping of workgroup memory, per-invocation
//! state snapshots, and barrier-based segment splitting.

use std::collections::BTreeMap;

use naga::{AddressSpace, Expression, Handle, Statement};

use crate::error::{NagaExecError, Result};
use crate::eval::type_byte_size;
use crate::value::Value;

/// Workgroup-scoped shared memory.
///
/// Allocated once per workgroup dispatch, shared across all invocations.
/// Keyed by `GlobalVariable` handle index for O(1) lookup.
#[derive(Default)]
pub(crate) struct WorkgroupMemory {
    buffers: BTreeMap<usize, Vec<u8>>,
}

impl WorkgroupMemory {
    pub(crate) fn new(module: &naga::Module) -> Self {
        let mut buffers = BTreeMap::new();
        for (handle, gv) in module.global_variables.iter() {
            if gv.space == AddressSpace::WorkGroup {
                let size = type_byte_size(module, &module.types[gv.ty].inner);
                buffers.insert(handle.index(), vec![0u8; size]);
            }
        }
        Self { buffers }
    }

    pub(crate) fn read(&self, handle_idx: usize, offset: usize, len: usize) -> &[u8] {
        let buf = &self.buffers[&handle_idx];
        &buf[offset..offset + len]
    }

    pub(crate) fn get_mut(&mut self, handle_idx: usize) -> Result<&mut Vec<u8>> {
        self.buffers.get_mut(&handle_idx).ok_or_else(|| {
            NagaExecError::TypeMismatch(format!("workgroup var handle {handle_idx} not allocated"))
        })
    }

    pub(crate) fn write(&mut self, handle_idx: usize, offset: usize, data: &[u8]) -> Result<()> {
        let buf = self.get_mut(handle_idx)?;
        buf[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    pub(crate) fn atomic_add_u32(
        &mut self,
        handle_idx: usize,
        offset: usize,
        val: u32,
    ) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    pub(crate) fn atomic_max_u32(
        &mut self,
        handle_idx: usize,
        offset: usize,
        val: u32,
    ) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.max(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    pub(crate) fn atomic_min_u32(
        &mut self,
        handle_idx: usize,
        offset: usize,
        val: u32,
    ) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.min(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    pub(crate) fn atomic_add_i32(
        &mut self,
        handle_idx: usize,
        offset: usize,
        val: i32,
    ) -> Result<i32> {
        let buf = self.get_mut(handle_idx)?;
        let old = i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    pub(crate) fn atomic_load_u32(&self, handle_idx: usize, offset: usize) -> Result<u32> {
        let buf = self.buffers.get(&handle_idx).ok_or_else(|| {
            NagaExecError::TypeMismatch(format!("workgroup var handle {handle_idx} not allocated"))
        })?;
        Ok(u32::from_le_bytes(
            buf[offset..offset + 4].try_into().unwrap_or([0; 4]),
        ))
    }

    pub(crate) fn atomic_store_u32(
        &mut self,
        handle_idx: usize,
        offset: usize,
        val: u32,
    ) -> Result<()> {
        let buf = self.get_mut(handle_idx)?;
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }
}

/// Saved per-invocation state for barrier-aware execution.
pub(crate) struct InvocationState {
    pub(crate) expr_cache: BTreeMap<Handle<Expression>, Value>,
    pub(crate) local_vars: BTreeMap<Handle<naga::LocalVariable>, Value>,
}

impl InvocationState {
    pub(crate) fn new() -> Self {
        Self {
            expr_cache: BTreeMap::new(),
            local_vars: BTreeMap::new(),
        }
    }
}

/// Split a block at top-level `Barrier` statements into segments.
///
/// Each segment is a `Vec<Statement>` that runs for all invocations before
/// the next segment begins. Barriers inside loops are handled by the loop
/// body being its own recursive execution scope.
pub(crate) fn split_at_barriers(body: &naga::Block) -> Vec<naga::Block> {
    let mut segments = Vec::new();
    let mut current = Vec::new();

    for stmt in body {
        if matches!(
            stmt,
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_)
        ) {
            segments.push(naga::Block::from_vec(current));
            current = Vec::new();
        } else {
            current.push(stmt.clone());
        }
    }
    if !current.is_empty() {
        segments.push(naga::Block::from_vec(current));
    }
    if segments.is_empty() {
        segments.push(naga::Block::default());
    }
    segments
}
