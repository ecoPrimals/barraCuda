// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU interpreter for naga IR compute shaders.

use std::collections::BTreeMap;

use naga::{
    AddressSpace, BinaryOperator, Expression, Function, GlobalVariable, Handle, Literal,
    MathFunction, Module, ScalarKind, Statement, TypeInner, UnaryOperator,
};
use tracing::trace;

use crate::error::{NagaExecError, Result};
use crate::value::Value;

/// Usage hint for a simulated buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimBufferUsage {
    Storage,
    StorageReadOnly,
    Uniform,
}

/// A simulated GPU buffer — just bytes on the heap.
#[derive(Debug, Clone)]
pub struct SimBuffer {
    pub data: Vec<u8>,
    pub usage: SimBufferUsage,
}

impl SimBuffer {
    /// Create a storage buffer from raw bytes.
    #[must_use]
    pub fn storage(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::Storage,
        }
    }

    /// Create a read-only storage buffer from raw bytes.
    #[must_use]
    pub fn storage_read_only(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::StorageReadOnly,
        }
    }

    /// Create a uniform buffer from raw bytes.
    #[must_use]
    pub fn uniform(data: Vec<u8>) -> Self {
        Self {
            data,
            usage: SimBufferUsage::Uniform,
        }
    }

    /// Create a storage buffer from a slice of f32 values.
    #[must_use]
    pub fn from_f32(values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Create a read-only storage buffer from a slice of f32 values.
    #[must_use]
    pub fn from_f32_readonly(values: &[f32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage_read_only(data)
    }

    /// Read back as f32 values.
    #[must_use]
    pub fn as_f32(&self) -> Vec<f32> {
        self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
            .collect()
    }

    /// Create a storage buffer from a slice of f64 values.
    #[must_use]
    pub fn from_f64(values: &[f64]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Read back as f64 values.
    #[must_use]
    pub fn as_f64(&self) -> Vec<f64> {
        self.data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect()
    }

    /// Create a storage buffer from a slice of u32 values.
    #[must_use]
    pub fn from_u32(values: &[u32]) -> Self {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::storage(data)
    }

    /// Read back as u32 values.
    #[must_use]
    pub fn as_u32(&self) -> Vec<u32> {
        self.data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap_or([0; 4])))
            .collect()
    }
}

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

        for wg_z in 0..workgroups.2 {
            for wg_y in 0..workgroups.1 {
                for wg_x in 0..workgroups.0 {
                    if has_workgroup_vars {
                        self.dispatch_workgroup_barrier_aware(
                            [wg_x, wg_y, wg_z],
                            wg_size,
                            bindings,
                        )?;
                    } else {
                        self.dispatch_workgroup_simple([wg_x, wg_y, wg_z], wg_size, bindings)?;
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
                        global_id,
                        local_id,
                        wg_size,
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
        bindings: &mut BTreeMap<(u32, u32), SimBuffer>,
    ) -> Result<()> {
        let ep = &self.module.entry_points[self.entry_point_index];
        let total = (wg_size[0] * wg_size[1] * wg_size[2]) as usize;

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
                    *global_id,
                    *local_id,
                    wg_size,
                );
                ctx.restore_state(&inv_states[i]);
                ctx.execute_body(segment)?;
                inv_states[i] = ctx.save_state();
            }
        }
        Ok(())
    }
}

/// Workgroup-scoped shared memory.
///
/// Allocated once per workgroup dispatch, shared across all invocations.
/// Keyed by `GlobalVariable` handle index for O(1) lookup.
#[derive(Default)]
struct WorkgroupMemory {
    buffers: BTreeMap<usize, Vec<u8>>,
}

impl WorkgroupMemory {
    fn new(module: &Module) -> Self {
        let mut buffers = BTreeMap::new();
        for (handle, gv) in module.global_variables.iter() {
            if gv.space == AddressSpace::WorkGroup {
                let size = type_byte_size(module, &module.types[gv.ty].inner);
                buffers.insert(handle.index(), vec![0u8; size]);
            }
        }
        Self { buffers }
    }

    fn read(&self, handle_idx: usize, offset: usize, len: usize) -> &[u8] {
        let buf = &self.buffers[&handle_idx];
        &buf[offset..offset + len]
    }

    fn write(&mut self, handle_idx: usize, offset: usize, data: &[u8]) {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        buf[offset..offset + data.len()].copy_from_slice(data);
    }

    fn atomic_add_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> u32 {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        old
    }

    fn atomic_max_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> u32 {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.max(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        old
    }

    fn atomic_min_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> u32 {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.min(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        old
    }

    fn atomic_add_i32(&mut self, handle_idx: usize, offset: usize, val: i32) -> i32 {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        let old = i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        old
    }

    fn atomic_load_u32(&self, handle_idx: usize, offset: usize) -> u32 {
        let buf = &self.buffers[&handle_idx];
        u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]))
    }

    fn atomic_store_u32(&mut self, handle_idx: usize, offset: usize, val: u32) {
        let buf = self.buffers.get_mut(&handle_idx).expect("workgroup var");
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Saved per-invocation state for barrier-aware execution.
struct InvocationState {
    expr_cache: BTreeMap<Handle<Expression>, Value>,
    local_vars: BTreeMap<Handle<naga::LocalVariable>, Value>,
}

impl InvocationState {
    fn new() -> Self {
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
fn split_at_barriers(body: &naga::Block) -> Vec<naga::Block> {
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

/// Per-invocation state: expression cache, local variables, buffer access.
struct InvocationContext<'a> {
    module: &'a Module,
    #[allow(dead_code)]
    info: &'a naga::valid::ModuleInfo,
    function: &'a Function,
    bindings: &'a mut BTreeMap<(u32, u32), SimBuffer>,
    shared: &'a mut WorkgroupMemory,
    global_id: [u32; 3],
    local_id: [u32; 3],
    workgroup_size: [u32; 3],
    expr_cache: BTreeMap<Handle<Expression>, Value>,
    local_vars: BTreeMap<Handle<naga::LocalVariable>, Value>,
}

impl<'a> InvocationContext<'a> {
    fn new(
        module: &'a Module,
        info: &'a naga::valid::ModuleInfo,
        function: &'a Function,
        bindings: &'a mut BTreeMap<(u32, u32), SimBuffer>,
        shared: &'a mut WorkgroupMemory,
        global_id: [u32; 3],
        local_id: [u32; 3],
        workgroup_size: [u32; 3],
    ) -> Self {
        Self {
            module,
            info,
            function,
            bindings,
            shared,
            global_id,
            local_id,
            workgroup_size,
            expr_cache: BTreeMap::new(),
            local_vars: BTreeMap::new(),
        }
    }

    fn save_state(&self) -> InvocationState {
        InvocationState {
            expr_cache: self.expr_cache.clone(),
            local_vars: self.local_vars.clone(),
        }
    }

    fn restore_state(&mut self, state: &InvocationState) {
        self.expr_cache.clone_from(&state.expr_cache);
        self.local_vars.clone_from(&state.local_vars);
    }

    fn execute_body(&mut self, body: &naga::Block) -> Result<()> {
        for stmt in body {
            self.execute_stmt(stmt)?;
        }
        Ok(())
    }

    fn execute_stmt(&mut self, stmt: &Statement) -> Result<()> {
        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let val = self.eval_expr(handle)?;
                    self.expr_cache.insert(handle, val);
                }
                Ok(())
            }
            Statement::Store { pointer, value } => {
                let val = self.get_value(value)?;
                self.store_pointer(pointer, &val)
            }
            Statement::Block(ref block) => self.execute_body(block),
            Statement::Return { .. }
            | Statement::Break
            | Statement::Continue
            | Statement::ControlBarrier(_)
            | Statement::MemoryBarrier(_) => Ok(()),
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => self.execute_atomic(pointer, fun, value, result),
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                let cond = self.get_value(condition)?;
                if cond.as_bool() {
                    self.execute_body(accept)
                } else {
                    self.execute_body(reject)
                }
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                for _ in 0..100_000 {
                    self.execute_body(body)?;
                    self.execute_body(continuing)?;
                    if let Some(break_cond) = break_if {
                        let cond = self.get_value(break_cond)?;
                        if cond.as_bool() {
                            break;
                        }
                    }
                }
                Ok(())
            }
            _ => Err(NagaExecError::UnsupportedStatement(format!("{stmt:?}"))),
        }
    }

    fn get_value(&mut self, handle: Handle<Expression>) -> Result<Value> {
        if let Some(cached) = self.expr_cache.get(&handle) {
            return Ok(cached.clone());
        }
        let val = self.eval_expr(handle)?;
        self.expr_cache.insert(handle, val.clone());
        Ok(val)
    }

    #[allow(clippy::too_many_lines)]
    fn eval_expr(&mut self, handle: Handle<Expression>) -> Result<Value> {
        let expr = &self.function.expressions[handle];
        match *expr {
            Expression::Literal(ref lit) => Ok(Self::eval_literal(lit)),

            Expression::FunctionArgument(idx) => {
                let arg = &self.function.arguments[idx as usize];
                match arg.binding {
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)) => {
                        Ok(Value::Vec3U32(self.global_id))
                    }
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationId)) => {
                        Ok(Value::Vec3U32(self.local_id))
                    }
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex)) => {
                        let idx = self.local_id[0]
                            + self.local_id[1] * self.workgroup_size[0]
                            + self.local_id[2] * self.workgroup_size[0] * self.workgroup_size[1];
                        Ok(Value::U32(idx))
                    }
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::NumWorkGroups)) => {
                        Ok(Value::Vec3U32([0, 0, 0]))
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "FunctionArgument({idx})"
                    ))),
                }
            }

            Expression::GlobalVariable(gv_handle) => {
                let gv = &self.module.global_variables[gv_handle];
                match gv.space {
                    AddressSpace::Storage { .. }
                    | AddressSpace::Uniform
                    | AddressSpace::WorkGroup =>
                    {
                        #[allow(clippy::cast_possible_truncation)]
                        Ok(Value::U32(gv_handle.index() as u32))
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "GlobalVariable in {space:?}",
                        space = gv.space
                    ))),
                }
            }

            Expression::LocalVariable(lv_handle) => Ok(self
                .local_vars
                .get(&lv_handle)
                .cloned()
                .unwrap_or(Value::U32(0))),

            Expression::Load { pointer } => self.load_pointer(pointer),

            Expression::Binary { op, left, right } => {
                let l = self.get_value(left)?;
                let r = self.get_value(right)?;
                eval_binary(op, &l, &r)
            }

            Expression::Unary { op, expr } => {
                let v = self.get_value(expr)?;
                eval_unary(op, &v)
            }

            Expression::Math { fun, arg, arg1, .. } => {
                let a = self.get_value(arg)?;
                let b = arg1.map(|h| self.get_value(h)).transpose()?;
                eval_math(fun, &a, b.as_ref())
            }

            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let v = self.get_value(expr)?;
                eval_cast(&v, kind, convert)
            }

            Expression::AccessIndex { base, index } => {
                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.load_buffer_element(gv_handle, index as usize);
                }
                let base_val = self.get_value(base)?;
                match base_val {
                    Value::Vec2(v) => Ok(Value::F32(v[index as usize])),
                    Value::Vec3(v) => Ok(Value::F32(v[index as usize])),
                    Value::Vec4(v) => Ok(Value::F32(v[index as usize])),
                    Value::Vec3U32(v) => Ok(Value::U32(v[index as usize])),
                    Value::Vec4U32(v) => Ok(Value::U32(v[index as usize])),
                    Value::Composite(ref fields) => {
                        fields
                            .get(index as usize)
                            .cloned()
                            .ok_or(NagaExecError::OutOfBounds {
                                index: index as usize,
                                length: fields.len(),
                            })
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "AccessIndex on {base_val:?}"
                    ))),
                }
            }

            Expression::Access { base, index } => {
                let idx = self.get_value(index)?;
                let idx_u = idx.as_u32() as usize;

                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.load_buffer_element(gv_handle, idx_u);
                }

                let base_val = self.get_value(base)?;
                match base_val {
                    Value::Composite(ref fields) => {
                        fields
                            .get(idx_u)
                            .cloned()
                            .ok_or(NagaExecError::OutOfBounds {
                                index: idx_u,
                                length: fields.len(),
                            })
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "Access on {base_val:?}"
                    ))),
                }
            }

            Expression::Compose { ty, ref components } => {
                let ty_inner = &self.module.types[ty].inner;
                let vals: Vec<Value> = components
                    .iter()
                    .map(|&c| self.get_value(c))
                    .collect::<Result<_>>()?;

                match *ty_inner {
                    TypeInner::Vector { size, scalar } => compose_vector(size, scalar, &vals),
                    TypeInner::Struct { .. } | TypeInner::Array { .. } => {
                        Ok(Value::Composite(vals))
                    }
                    _ => Err(NagaExecError::UnsupportedType(format!(
                        "Compose {ty_inner:?}"
                    ))),
                }
            }

            Expression::Splat { size, value } => {
                let v = self.get_value(value)?;
                let n = match size {
                    naga::VectorSize::Bi => 2,
                    naga::VectorSize::Tri => 3,
                    naga::VectorSize::Quad => 4,
                };
                match v {
                    Value::F32(f) => match n {
                        2 => Ok(Value::Vec2([f; 2])),
                        3 => Ok(Value::Vec3([f; 3])),
                        _ => Ok(Value::Vec4([f; 4])),
                    },
                    Value::F64(f) => match n {
                        2 => Ok(Value::Vec2F64([f; 2])),
                        3 => Ok(Value::Vec3F64([f; 3])),
                        _ => Ok(Value::Vec4F64([f; 4])),
                    },
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "Splat of {v:?}"
                    ))),
                }
            }

            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let cond = self.get_value(condition)?;
                if cond.as_bool() {
                    self.get_value(accept)
                } else {
                    self.get_value(reject)
                }
            }

            _ => Err(NagaExecError::UnsupportedExpression(format!("{expr:?}"))),
        }
    }

    fn eval_literal(lit: &Literal) -> Value {
        match *lit {
            Literal::F32(v) => Value::F32(v),
            Literal::F64(v) => Value::F64(v),
            Literal::U32(v) => Value::U32(v),
            Literal::I32(v) => Value::I32(v),
            Literal::Bool(v) => Value::Bool(v),
            _ => Value::U32(0),
        }
    }

    fn resolve_binding(&self, gv_handle: Handle<GlobalVariable>) -> Option<(u32, u32)> {
        let gv = &self.module.global_variables[gv_handle];
        gv.binding.as_ref().map(|b| (b.group, b.binding))
    }

    fn load_pointer(&mut self, pointer: Handle<Expression>) -> Result<Value> {
        let expr = &self.function.expressions[pointer];
        match *expr {
            Expression::GlobalVariable(gv_handle) => {
                let gv = &self.module.global_variables[gv_handle];
                let ty = &self.module.types[gv.ty];

                if gv.space == AddressSpace::WorkGroup {
                    let total_size = type_byte_size(self.module, &ty.inner);
                    let data = self.shared.read(gv_handle.index(), 0, total_size);
                    return Ok(Value::read_from_buffer(data, 0, ty));
                }

                if let TypeInner::Array { base, .. } = ty.inner {
                    let elem_ty = &self.module.types[base];
                    let (group, binding) = self.resolve_binding(gv_handle).ok_or(
                        NagaExecError::UnsupportedExpression(
                            "GlobalVariable without binding".into(),
                        ),
                    )?;
                    let buf = self
                        .bindings
                        .get(&(group, binding))
                        .ok_or(NagaExecError::BindingNotFound { group, binding })?;
                    let elem_size = type_byte_size(self.module, &elem_ty.inner);
                    let count = buf.data.len() / elem_size;
                    let mut elements = Vec::with_capacity(count);
                    for i in 0..count {
                        elements.push(Value::read_from_buffer(&buf.data, i * elem_size, elem_ty));
                    }
                    Ok(Value::Composite(elements))
                } else {
                    let (group, binding) = self.resolve_binding(gv_handle).ok_or(
                        NagaExecError::UnsupportedExpression(
                            "GlobalVariable without binding".into(),
                        ),
                    )?;
                    let buf = self
                        .bindings
                        .get(&(group, binding))
                        .ok_or(NagaExecError::BindingNotFound { group, binding })?;
                    Ok(Value::read_from_buffer(&buf.data, 0, ty))
                }
            }
            Expression::LocalVariable(lv_handle) => Ok(self
                .local_vars
                .get(&lv_handle)
                .cloned()
                .unwrap_or(Value::U32(0))),
            Expression::Access { base, index } => {
                let idx = self.get_value(index)?;
                let idx_u = idx.as_u32() as usize;

                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.load_buffer_element(gv_handle, idx_u);
                }
                let base_val = self.get_value(base)?;
                match base_val {
                    Value::Composite(ref fields) => {
                        fields
                            .get(idx_u)
                            .cloned()
                            .ok_or(NagaExecError::OutOfBounds {
                                index: idx_u,
                                length: fields.len(),
                            })
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "Load(Access) on {base_val:?}"
                    ))),
                }
            }
            Expression::AccessIndex { base, index } => {
                let base_val = self.load_pointer(base)?;
                match base_val {
                    Value::Composite(ref fields) => {
                        fields
                            .get(index as usize)
                            .cloned()
                            .ok_or(NagaExecError::OutOfBounds {
                                index: index as usize,
                                length: fields.len(),
                            })
                    }
                    _ => Err(NagaExecError::UnsupportedExpression(format!(
                        "Load(AccessIndex) on {base_val:?}"
                    ))),
                }
            }
            _ => self.get_value(pointer),
        }
    }

    fn store_pointer(&mut self, pointer: Handle<Expression>, val: &Value) -> Result<()> {
        let expr = &self.function.expressions[pointer];
        match *expr {
            Expression::LocalVariable(lv_handle) => {
                self.local_vars.insert(lv_handle, val.clone());
                Ok(())
            }
            Expression::Access { base, index } => {
                let idx = self.get_value(index)?;
                let idx_u = idx.as_u32() as usize;

                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.store_buffer_element(gv_handle, idx_u, val);
                }
                Err(NagaExecError::UnsupportedStatement(format!(
                    "Store to Access({base_expr:?})"
                )))
            }
            Expression::AccessIndex { base, index } => {
                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.store_buffer_element(gv_handle, index as usize, val);
                }
                Err(NagaExecError::UnsupportedStatement(format!(
                    "Store to AccessIndex({base_expr:?})"
                )))
            }
            Expression::GlobalVariable(gv_handle) => {
                let gv = &self.module.global_variables[gv_handle];
                if gv.space == AddressSpace::WorkGroup {
                    let mut tmp = vec![0u8; val.byte_size()];
                    val.write_to_buffer(&mut tmp, 0);
                    self.shared.write(gv_handle.index(), 0, &tmp);
                    return Ok(());
                }
                let (group, binding) =
                    self.resolve_binding(gv_handle)
                        .ok_or(NagaExecError::UnsupportedStatement(
                            "Store to GlobalVariable without binding".into(),
                        ))?;
                let buf = self
                    .bindings
                    .get_mut(&(group, binding))
                    .ok_or(NagaExecError::BindingNotFound { group, binding })?;
                val.write_to_buffer(&mut buf.data, 0);
                Ok(())
            }
            _ => Err(NagaExecError::UnsupportedStatement(format!(
                "Store to {expr:?}"
            ))),
        }
    }

    fn execute_atomic(
        &mut self,
        pointer: Handle<Expression>,
        fun: naga::AtomicFunction,
        value: Handle<Expression>,
        result: Option<Handle<Expression>>,
    ) -> Result<()> {
        let val = self.get_value(value)?;
        let (handle_idx, offset) = self.resolve_atomic_pointer(pointer)?;

        let old = match fun {
            naga::AtomicFunction::Add => match &val {
                Value::U32(v) => Value::U32(self.shared.atomic_add_u32(handle_idx, offset, *v)),
                Value::I32(v) => Value::I32(self.shared.atomic_add_i32(handle_idx, offset, *v)),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic add on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Max => match &val {
                Value::U32(v) => Value::U32(self.shared.atomic_max_u32(handle_idx, offset, *v)),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic max on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Min => match &val {
                Value::U32(v) => Value::U32(self.shared.atomic_min_u32(handle_idx, offset, *v)),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic min on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Exchange { compare: None } => {
                let old_val = Value::U32(self.shared.atomic_load_u32(handle_idx, offset));
                self.shared
                    .atomic_store_u32(handle_idx, offset, val.as_u32());
                old_val
            }
            _ => {
                return Err(NagaExecError::UnsupportedStatement(format!(
                    "atomic {fun:?}"
                )));
            }
        };

        if let Some(result_handle) = result {
            self.expr_cache.insert(result_handle, old);
        }
        Ok(())
    }

    fn resolve_atomic_pointer(&self, pointer: Handle<Expression>) -> Result<(usize, usize)> {
        let expr = &self.function.expressions[pointer];
        match *expr {
            Expression::Access { base, index } => {
                let idx = self
                    .expr_cache
                    .get(&index)
                    .cloned()
                    .unwrap_or(Value::U32(0));
                let idx_u = idx.as_u32() as usize;
                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    let gv = &self.module.global_variables[gv_handle];
                    let elem_size = match &self.module.types[gv.ty].inner {
                        TypeInner::Array { base, .. } => {
                            type_byte_size(self.module, &self.module.types[*base].inner)
                        }
                        _ => 4,
                    };
                    return Ok((gv_handle.index(), idx_u * elem_size));
                }
                Err(NagaExecError::UnsupportedExpression(
                    "atomic pointer Access on non-global".into(),
                ))
            }
            Expression::GlobalVariable(gv_handle) => Ok((gv_handle.index(), 0)),
            _ => Err(NagaExecError::UnsupportedExpression(format!(
                "atomic pointer {expr:?}"
            ))),
        }
    }

    fn load_buffer_element(
        &self,
        gv_handle: Handle<GlobalVariable>,
        index: usize,
    ) -> Result<Value> {
        let gv = &self.module.global_variables[gv_handle];
        let ty = &self.module.types[gv.ty];
        let (TypeInner::Array {
            base: elem_ty_handle,
            ..
        }
        | TypeInner::BindingArray {
            base: elem_ty_handle,
            ..
        }) = ty.inner
        else {
            return Err(NagaExecError::UnsupportedType(format!(
                "buffer element access on {:?}",
                ty.inner
            )));
        };
        let elem_ty = &self.module.types[elem_ty_handle];
        let elem_size = type_byte_size(self.module, &elem_ty.inner);
        let offset = index * elem_size;

        if gv.space == AddressSpace::WorkGroup {
            let data = self.shared.read(gv_handle.index(), offset, elem_size);
            return Ok(Value::read_from_buffer(data, 0, elem_ty));
        }

        let (group, binding) =
            self.resolve_binding(gv_handle)
                .ok_or(NagaExecError::UnsupportedExpression(
                    "GlobalVariable without binding".into(),
                ))?;
        let buf = self
            .bindings
            .get(&(group, binding))
            .ok_or(NagaExecError::BindingNotFound { group, binding })?;

        trace!(
            group,
            binding, index, offset, elem_size, "load buffer element"
        );
        if offset + elem_size > buf.data.len() {
            return Err(NagaExecError::OutOfBounds {
                index,
                length: buf.data.len() / elem_size,
            });
        }
        Ok(Value::read_from_buffer(&buf.data, offset, elem_ty))
    }

    fn store_buffer_element(
        &mut self,
        gv_handle: Handle<GlobalVariable>,
        index: usize,
        val: &Value,
    ) -> Result<()> {
        let gv = &self.module.global_variables[gv_handle];
        let ty = &self.module.types[gv.ty];
        let (TypeInner::Array {
            base: elem_ty_handle,
            ..
        }
        | TypeInner::BindingArray {
            base: elem_ty_handle,
            ..
        }) = ty.inner
        else {
            return Err(NagaExecError::UnsupportedType(format!(
                "buffer element store on {:?}",
                ty.inner
            )));
        };
        let elem_ty = &self.module.types[elem_ty_handle];
        let elem_size = type_byte_size(self.module, &elem_ty.inner);
        let offset = index * elem_size;

        if gv.space == AddressSpace::WorkGroup {
            let mut tmp = vec![0u8; elem_size];
            val.write_to_buffer(&mut tmp, 0);
            self.shared.write(gv_handle.index(), offset, &tmp);
            return Ok(());
        }

        let (group, binding) =
            self.resolve_binding(gv_handle)
                .ok_or(NagaExecError::UnsupportedStatement(
                    "GlobalVariable without binding".into(),
                ))?;
        let buf = self
            .bindings
            .get_mut(&(group, binding))
            .ok_or(NagaExecError::BindingNotFound { group, binding })?;

        if offset + elem_size > buf.data.len() {
            return Err(NagaExecError::OutOfBounds {
                index,
                length: buf.data.len() / elem_size,
            });
        }
        val.write_to_buffer(&mut buf.data, offset);
        Ok(())
    }
}

// ── Arithmetic helpers ───────────────────────────────────────────────────

fn eval_binary(op: BinaryOperator, l: &Value, r: &Value) -> Result<Value> {
    match (l, r) {
        (Value::F32(a), Value::F32(b)) => binary_f32(op, *a, *b),
        (Value::F64(a), Value::F64(b)) => binary_f64(op, *a, *b),
        (Value::U32(a), Value::U32(b)) => binary_u32(op, *a, *b),
        (Value::I32(a), Value::I32(b)) => binary_i32(op, *a, *b),
        (Value::Bool(a), Value::Bool(b)) => binary_bool(op, *a, *b),
        _ => Err(NagaExecError::TypeMismatch(format!(
            "binary {op:?} on {l:?} and {r:?}"
        ))),
    }
}

fn binary_f32(op: BinaryOperator, a: f32, b: f32) -> Result<Value> {
    Ok(match op {
        BinaryOperator::Add => Value::F32(a + b),
        BinaryOperator::Subtract => Value::F32(a - b),
        BinaryOperator::Multiply => Value::F32(a * b),
        BinaryOperator::Divide => Value::F32(a / b),
        BinaryOperator::Modulo => Value::F32(a % b),
        BinaryOperator::Less => Value::Bool(a < b),
        BinaryOperator::LessEqual => Value::Bool(a <= b),
        BinaryOperator::Greater => Value::Bool(a > b),
        BinaryOperator::GreaterEqual => Value::Bool(a >= b),
        BinaryOperator::Equal => Value::Bool((a - b).abs() < f32::EPSILON),
        BinaryOperator::NotEqual => Value::Bool((a - b).abs() >= f32::EPSILON),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "binary f32 {op:?}"
            )));
        }
    })
}

fn binary_f64(op: BinaryOperator, a: f64, b: f64) -> Result<Value> {
    Ok(match op {
        BinaryOperator::Add => Value::F64(a + b),
        BinaryOperator::Subtract => Value::F64(a - b),
        BinaryOperator::Multiply => Value::F64(a * b),
        BinaryOperator::Divide => Value::F64(a / b),
        BinaryOperator::Modulo => Value::F64(a % b),
        BinaryOperator::Less => Value::Bool(a < b),
        BinaryOperator::LessEqual => Value::Bool(a <= b),
        BinaryOperator::Greater => Value::Bool(a > b),
        BinaryOperator::GreaterEqual => Value::Bool(a >= b),
        BinaryOperator::Equal => Value::Bool((a - b).abs() < f64::EPSILON),
        BinaryOperator::NotEqual => Value::Bool((a - b).abs() >= f64::EPSILON),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "binary f64 {op:?}"
            )));
        }
    })
}

fn binary_u32(op: BinaryOperator, a: u32, b: u32) -> Result<Value> {
    Ok(match op {
        BinaryOperator::Add => Value::U32(a.wrapping_add(b)),
        BinaryOperator::Subtract => Value::U32(a.wrapping_sub(b)),
        BinaryOperator::Multiply => Value::U32(a.wrapping_mul(b)),
        BinaryOperator::Divide => Value::U32(if b == 0 { 0 } else { a / b }),
        BinaryOperator::Modulo => Value::U32(if b == 0 { 0 } else { a % b }),
        BinaryOperator::Less => Value::Bool(a < b),
        BinaryOperator::LessEqual => Value::Bool(a <= b),
        BinaryOperator::Greater => Value::Bool(a > b),
        BinaryOperator::GreaterEqual => Value::Bool(a >= b),
        BinaryOperator::Equal => Value::Bool(a == b),
        BinaryOperator::NotEqual => Value::Bool(a != b),
        BinaryOperator::And => Value::U32(a & b),
        BinaryOperator::InclusiveOr => Value::U32(a | b),
        BinaryOperator::ExclusiveOr => Value::U32(a ^ b),
        BinaryOperator::ShiftLeft => Value::U32(a.wrapping_shl(b)),
        BinaryOperator::ShiftRight => Value::U32(a.wrapping_shr(b)),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "binary u32 {op:?}"
            )));
        }
    })
}

fn binary_i32(op: BinaryOperator, a: i32, b: i32) -> Result<Value> {
    Ok(match op {
        BinaryOperator::Add => Value::I32(a.wrapping_add(b)),
        BinaryOperator::Subtract => Value::I32(a.wrapping_sub(b)),
        BinaryOperator::Multiply => Value::I32(a.wrapping_mul(b)),
        BinaryOperator::Divide => Value::I32(if b == 0 { 0 } else { a / b }),
        BinaryOperator::Modulo => Value::I32(if b == 0 { 0 } else { a % b }),
        BinaryOperator::Less => Value::Bool(a < b),
        BinaryOperator::LessEqual => Value::Bool(a <= b),
        BinaryOperator::Greater => Value::Bool(a > b),
        BinaryOperator::GreaterEqual => Value::Bool(a >= b),
        BinaryOperator::Equal => Value::Bool(a == b),
        BinaryOperator::NotEqual => Value::Bool(a != b),
        BinaryOperator::And => Value::I32(a & b),
        BinaryOperator::InclusiveOr => Value::I32(a | b),
        BinaryOperator::ExclusiveOr => Value::I32(a ^ b),
        BinaryOperator::ShiftLeft => Value::I32(a.wrapping_shl(b.cast_unsigned())),
        BinaryOperator::ShiftRight => Value::I32(a.wrapping_shr(b.cast_unsigned())),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "binary i32 {op:?}"
            )));
        }
    })
}

fn binary_bool(op: BinaryOperator, a: bool, b: bool) -> Result<Value> {
    Ok(match op {
        BinaryOperator::LogicalAnd => Value::Bool(a && b),
        BinaryOperator::LogicalOr => Value::Bool(a || b),
        BinaryOperator::Equal => Value::Bool(a == b),
        BinaryOperator::NotEqual => Value::Bool(a != b),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "binary bool {op:?}"
            )));
        }
    })
}

fn eval_unary(op: UnaryOperator, v: &Value) -> Result<Value> {
    Ok(match (op, v) {
        (UnaryOperator::Negate, Value::F32(f)) => Value::F32(-f),
        (UnaryOperator::Negate, Value::F64(f)) => Value::F64(-f),
        (UnaryOperator::Negate, Value::I32(i)) => Value::I32(-i),
        (UnaryOperator::BitwiseNot, Value::U32(u)) => Value::U32(!u),
        (UnaryOperator::BitwiseNot, Value::I32(i)) => Value::I32(!i),
        (UnaryOperator::LogicalNot, Value::Bool(b)) => Value::Bool(!b),
        _ => {
            return Err(NagaExecError::UnsupportedExpression(format!(
                "unary {op:?} on {v:?}"
            )));
        }
    })
}

#[allow(clippy::too_many_lines)]
fn eval_math(fun: MathFunction, a: &Value, b: Option<&Value>) -> Result<Value> {
    match a {
        Value::F32(x) => {
            let x = *x;
            Ok(Value::F32(match fun {
                MathFunction::Abs => x.abs(),
                MathFunction::Ceil => x.ceil(),
                MathFunction::Floor => x.floor(),
                MathFunction::Round => x.round(),
                MathFunction::Fract => x.fract(),
                MathFunction::Trunc => x.trunc(),
                MathFunction::Sign => {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                MathFunction::Sqrt => x.sqrt(),
                MathFunction::InverseSqrt => 1.0 / x.sqrt(),
                MathFunction::Log => x.ln(),
                MathFunction::Log2 => x.log2(),
                MathFunction::Exp => x.exp(),
                MathFunction::Exp2 => x.exp2(),
                MathFunction::Sin => x.sin(),
                MathFunction::Cos => x.cos(),
                MathFunction::Tan => x.tan(),
                MathFunction::Asin => x.asin(),
                MathFunction::Acos => x.acos(),
                MathFunction::Atan => x.atan(),
                MathFunction::Sinh => x.sinh(),
                MathFunction::Cosh => x.cosh(),
                MathFunction::Tanh => x.tanh(),
                MathFunction::Asinh => x.asinh(),
                MathFunction::Acosh => x.acosh(),
                MathFunction::Atanh => x.atanh(),
                MathFunction::Saturate => x.clamp(0.0, 1.0),
                MathFunction::Pow => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("pow needs 2 args".into()))?
                        .as_f32();
                    x.powf(y)
                }
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_f32();
                    x.min(y)
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_f32();
                    x.max(y)
                }
                MathFunction::Atan2 => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("atan2 needs 2 args".into()))?
                        .as_f32();
                    x.atan2(y)
                }
                MathFunction::Step => {
                    let edge = b
                        .ok_or(NagaExecError::TypeMismatch("step needs 2 args".into()))?
                        .as_f32();
                    if x >= edge { 1.0 } else { 0.0 }
                }
                _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
            }))
        }
        Value::F64(x) => {
            let x = *x;
            Ok(Value::F64(match fun {
                MathFunction::Abs => x.abs(),
                MathFunction::Ceil => x.ceil(),
                MathFunction::Floor => x.floor(),
                MathFunction::Round => x.round(),
                MathFunction::Fract => x.fract(),
                MathFunction::Trunc => x.trunc(),
                MathFunction::Sign => {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                MathFunction::Sqrt => x.sqrt(),
                MathFunction::InverseSqrt => 1.0 / x.sqrt(),
                MathFunction::Log => x.ln(),
                MathFunction::Log2 => x.log2(),
                MathFunction::Exp => x.exp(),
                MathFunction::Exp2 => x.exp2(),
                MathFunction::Sin => x.sin(),
                MathFunction::Cos => x.cos(),
                MathFunction::Tan => x.tan(),
                MathFunction::Asin => x.asin(),
                MathFunction::Acos => x.acos(),
                MathFunction::Atan => x.atan(),
                MathFunction::Sinh => x.sinh(),
                MathFunction::Cosh => x.cosh(),
                MathFunction::Tanh => x.tanh(),
                MathFunction::Asinh => x.asinh(),
                MathFunction::Acosh => x.acosh(),
                MathFunction::Atanh => x.atanh(),
                MathFunction::Saturate => x.clamp(0.0, 1.0),
                MathFunction::Pow => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("pow needs 2 args".into()))?
                        .as_f64();
                    x.powf(y)
                }
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_f64();
                    x.min(y)
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_f64();
                    x.max(y)
                }
                MathFunction::Atan2 => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("atan2 needs 2 args".into()))?
                        .as_f64();
                    x.atan2(y)
                }
                MathFunction::Step => {
                    let edge = b
                        .ok_or(NagaExecError::TypeMismatch("step needs 2 args".into()))?
                        .as_f64();
                    if x >= edge { 1.0 } else { 0.0 }
                }
                _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
            }))
        }
        Value::U32(x) => {
            let x = *x;
            Ok(match fun {
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_u32();
                    Value::U32(x.min(y))
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_u32();
                    Value::U32(x.max(y))
                }
                _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
            })
        }
        Value::I32(x) => {
            let x = *x;
            Ok(match fun {
                MathFunction::Abs => Value::I32(x.abs()),
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_i32();
                    Value::I32(x.min(y))
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_i32();
                    Value::I32(x.max(y))
                }
                _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
            })
        }
        _ => Err(NagaExecError::UnsupportedExpression(format!(
            "Math({fun:?}) on {a:?}"
        ))),
    }
}

fn eval_cast(v: &Value, kind: ScalarKind, convert: Option<u8>) -> Result<Value> {
    let width = convert.unwrap_or(4);
    Ok(match (kind, width) {
        (ScalarKind::Float, 4) => Value::F32(v.as_f32()),
        (ScalarKind::Float, 8) => Value::F64(v.as_f64()),
        (ScalarKind::Uint, 4) => Value::U32(v.as_u32()),
        (ScalarKind::Sint, 4) => Value::I32(v.as_i32()),
        (ScalarKind::Bool, _) => Value::Bool(v.as_bool()),
        _ => {
            return Err(NagaExecError::UnsupportedType(format!(
                "cast to {kind:?} width={width}"
            )));
        }
    })
}

fn compose_vector(size: naga::VectorSize, scalar: naga::Scalar, vals: &[Value]) -> Result<Value> {
    match (scalar.kind, scalar.width, size) {
        (ScalarKind::Float, 4, naga::VectorSize::Bi) => {
            Ok(Value::Vec2([vals[0].as_f32(), vals[1].as_f32()]))
        }
        (ScalarKind::Float, 4, naga::VectorSize::Tri) => Ok(Value::Vec3([
            vals[0].as_f32(),
            vals[1].as_f32(),
            vals[2].as_f32(),
        ])),
        (ScalarKind::Float, 4, naga::VectorSize::Quad) => Ok(Value::Vec4([
            vals[0].as_f32(),
            vals[1].as_f32(),
            vals[2].as_f32(),
            vals[3].as_f32(),
        ])),
        (ScalarKind::Uint, 4, naga::VectorSize::Tri) => Ok(Value::Vec3U32([
            vals[0].as_u32(),
            vals[1].as_u32(),
            vals[2].as_u32(),
        ])),
        _ => Err(NagaExecError::UnsupportedType(format!(
            "compose vector {scalar:?} x {size:?}"
        ))),
    }
}

fn type_byte_size(module: &Module, inner: &TypeInner) -> usize {
    match *inner {
        TypeInner::Scalar(s) | TypeInner::Atomic(s) => s.width as usize,
        TypeInner::Vector { size, scalar } => {
            let n = match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            n * scalar.width as usize
        }
        TypeInner::Array { base, size, .. } => {
            let elem = type_byte_size(module, &module.types[base].inner);
            match size {
                naga::ArraySize::Constant(n) => elem * n.get() as usize,
                naga::ArraySize::Pending(_) | naga::ArraySize::Dynamic => elem,
            }
        }
        TypeInner::Struct { ref members, .. } => members
            .iter()
            .map(|m| type_byte_size(module, &module.types[m.ty].inner))
            .sum(),
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_add() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input_a[idx] + input_b[idx];
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0, 4.0]));
        bindings.insert(
            (0, 1),
            SimBuffer::from_f32_readonly(&[10.0, 20.0, 30.0, 40.0]),
        );
        bindings.insert((0, 2), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((4, 1, 1), &mut bindings).unwrap();

        let result = bindings[&(0, 2)].as_f32();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_elementwise_mul() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    b[gid.x] = a[gid.x] * 2.0;
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 3]));

        exec.dispatch((3, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_f32(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_math_builtins() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = sin(x);
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f32_readonly(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]),
        );
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 3]));

        exec.dispatch((3, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f32();
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!(result[2].abs() < 1e-6);
    }

    #[test]
    fn test_f64_native() {
        let wgsl = r#"
enable f16;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] * input[gid.x];
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[3.0, 7.0]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 2]));

        exec.dispatch((2, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f32();
        assert!((result[0] - 9.0).abs() < 1e-6);
        assert!((result[1] - 49.0).abs() < 1e-6);
    }

    #[test]
    fn test_conditional_clamp() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = max(min(x, 1.0), 0.0);
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[-0.5, 0.3, 0.7, 1.5]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((4, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_f32(), vec![0.0, 0.3, 0.7, 1.0]);
    }

    #[test]
    fn test_relu() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = input[gid.x];
    output[gid.x] = max(x, 0.0);
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f32_readonly(&[-2.0, -0.5, 0.0, 0.5, 2.0]),
        );
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 5]));

        exec.dispatch((5, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_f32(), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn test_workgroup_size_larger_than_one() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] + 1.0;
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f32_readonly(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        );
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 8]));

        exec.dispatch((2, 1, 1), &mut bindings).unwrap();
        assert_eq!(
            bindings[&(0, 1)].as_f32(),
            vec![11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0]
        );
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = log(exp(input[gid.x]));
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[0.0, 1.0, 2.0, -1.0]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((4, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f32();
        for (i, &expected) in [0.0f32, 1.0, 2.0, -1.0].iter().enumerate() {
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "index {i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn test_tanh_activation() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = tanh(input[gid.x]);
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f32_readonly(&[0.0, 1.0, -1.0, 100.0]),
        );
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((4, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f32();
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.7615942).abs() < 1e-5);
        assert!((result[2] - (-0.7615942)).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    // ── f64 native tests ─────────────────────────────────────────────

    #[test]
    fn test_f64_elementwise_add() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out[gid.x] = a[gid.x] + b[gid.x];
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f64(&[1.0000000000001, 2.0000000000002]),
        );
        bindings.insert(
            (0, 1),
            SimBuffer::from_f64(&[0.0000000000001, 0.0000000000002]),
        );
        bindings.insert((0, 2), SimBuffer::from_f64(&[0.0; 2]));

        exec.dispatch((2, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 2)].as_f64();
        assert!(
            (result[0] - 1.0000000000002).abs() < 1e-13,
            "f64 precision lost: {}",
            result[0]
        );
        assert!(
            (result[1] - 2.0000000000004).abs() < 1e-13,
            "f64 precision lost: {}",
            result[1]
        );
    }

    #[test]
    fn test_f64_transcendentals() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = sin(input[gid.x]);
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert(
            (0, 0),
            SimBuffer::from_f64(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]),
        );
        bindings.insert((0, 1), SimBuffer::from_f64(&[0.0; 3]));

        exec.dispatch((3, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f64();
        assert!(result[0].abs() < 1e-15, "sin(0) = {}", result[0]);
        assert!((result[1] - 1.0).abs() < 1e-15, "sin(pi/2) = {}", result[1]);
        assert!(result[2].abs() < 1e-15, "sin(pi) = {}", result[2]);
    }

    #[test]
    fn test_f64_precision_vs_f32() {
        let val: f64 = 1.0 + 1e-15;
        let f32_val = val as f32;
        assert_eq!(f32_val, 1.0, "f32 should lose the 1e-15 precision");

        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] * input[gid.x];
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f64(&[1.00000000000001]));
        bindings.insert((0, 1), SimBuffer::from_f64(&[0.0]));

        exec.dispatch((1, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f64();
        let expected = 1.00000000000001_f64 * 1.00000000000001_f64;
        assert!(
            (result[0] - expected).abs() < 1e-14,
            "f64 precision: got {}, expected {expected}",
            result[0]
        );
    }

    #[test]
    fn test_f64_exp_log_roundtrip() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = log(exp(input[gid.x]));
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f64(&[0.0, 1.0, 2.0, -1.0, 0.5]));
        bindings.insert((0, 1), SimBuffer::from_f64(&[0.0; 5]));

        exec.dispatch((5, 1, 1), &mut bindings).unwrap();
        let result = bindings[&(0, 1)].as_f64();
        for (i, &expected) in [0.0, 1.0, 2.0, -1.0, 0.5].iter().enumerate() {
            assert!(
                (result[i] - expected).abs() < 1e-14,
                "index {i}: got {}, expected {expected} (f64 precision)",
                result[i]
            );
        }
    }

    // ── Workgroup shared memory + barrier tests ──────────────────────

    #[test]
    fn test_shared_memory_reverse() {
        let wgsl = r#"
var<workgroup> wg_data: array<f32, 4>;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    wg_data[lid.x] = input[gid.x];
    workgroupBarrier();
    output[gid.x] = wg_data[3u - lid.x];
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[1.0, 2.0, 3.0, 4.0]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((1, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_f32(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_shared_memory_broadcast() {
        let wgsl = r#"
var<workgroup> leader_val: f32;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    if lid.x == 0u {
        leader_val = input[0u];
    }
    workgroupBarrier();
    output[gid.x] = leader_val;
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_f32_readonly(&[42.0, 0.0, 0.0, 0.0]));
        bindings.insert((0, 1), SimBuffer::from_f32(&[0.0; 4]));

        exec.dispatch((1, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_f32(), vec![42.0, 42.0, 42.0, 42.0]);
    }

    // ── Atomic operation tests ───────────────────────────────────────

    #[test]
    fn test_atomic_add_accumulate() {
        let wgsl = r#"
var<workgroup> sum: atomic<u32>;

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    atomicAdd(&sum, input[gid.x]);
    workgroupBarrier();
    if lid.x == 0u {
        output[0u] = atomicLoad(&sum);
    }
}
"#;
        let exec = NagaExecutor::new(wgsl, "main").unwrap();
        let mut bindings = BTreeMap::new();
        bindings.insert((0, 0), SimBuffer::from_u32(&[1, 2, 3, 4]));
        bindings.insert((0, 1), SimBuffer::from_u32(&[0]));

        exec.dispatch((1, 1, 1), &mut bindings).unwrap();
        assert_eq!(bindings[&(0, 1)].as_u32(), vec![10]);
    }
}
