// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU interpreter for naga IR compute shaders.

use std::collections::BTreeMap;

use naga::{
    AddressSpace, Expression, Function, GlobalVariable, Handle, Literal, Module, Statement,
    TypeInner,
};
use tracing::trace;

use crate::error::{NagaExecError, Result};
use crate::eval::{compose_vector, eval_binary, eval_cast, eval_math, eval_unary, type_byte_size};
use crate::sim_buffer::SimBuffer;
use crate::value::Value;

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

    fn get_mut(&mut self, handle_idx: usize) -> Result<&mut Vec<u8>> {
        self.buffers
            .get_mut(&handle_idx)
            .ok_or(NagaExecError::TypeMismatch(format!(
                "workgroup var handle {handle_idx} not allocated"
            )))
    }

    fn write(&mut self, handle_idx: usize, offset: usize, data: &[u8]) -> Result<()> {
        let buf = self.get_mut(handle_idx)?;
        buf[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    fn atomic_add_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    fn atomic_max_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.max(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    fn atomic_min_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> Result<u32> {
        let buf = self.get_mut(handle_idx)?;
        let old = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.min(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    fn atomic_add_i32(&mut self, handle_idx: usize, offset: usize, val: i32) -> Result<i32> {
        let buf = self.get_mut(handle_idx)?;
        let old = i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let new = old.wrapping_add(val);
        buf[offset..offset + 4].copy_from_slice(&new.to_le_bytes());
        Ok(old)
    }

    fn atomic_load_u32(&self, handle_idx: usize, offset: usize) -> Result<u32> {
        let buf = self
            .buffers
            .get(&handle_idx)
            .ok_or(NagaExecError::TypeMismatch(format!(
                "workgroup var handle {handle_idx} not allocated"
            )))?;
        Ok(u32::from_le_bytes(
            buf[offset..offset + 4].try_into().unwrap_or([0; 4]),
        ))
    }

    fn atomic_store_u32(&mut self, handle_idx: usize, offset: usize, val: u32) -> Result<()> {
        let buf = self.get_mut(handle_idx)?;
        buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        Ok(())
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
    #[expect(
        dead_code,
        reason = "retained for future validation-aware interpretation"
    )]
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
    #[expect(
        clippy::too_many_arguments,
        reason = "invocation context requires all GPU execution state"
    )]
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
                if cond.as_bool()? {
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
                        if cond.as_bool()? {
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

    #[expect(
        clippy::too_many_lines,
        reason = "exhaustive naga Expression dispatch — splitting would obscure the interpretation flow"
    )]
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
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "naga global variable handle index fits u32"
                        )]
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
                let idx_u = idx.as_u32()? as usize;

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
                if cond.as_bool()? {
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
                let idx_u = idx.as_u32()? as usize;

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
                let idx_u = idx.as_u32()? as usize;

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
                    self.shared.write(gv_handle.index(), 0, &tmp)?;
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
                Value::U32(v) => Value::U32(self.shared.atomic_add_u32(handle_idx, offset, *v)?),
                Value::I32(v) => Value::I32(self.shared.atomic_add_i32(handle_idx, offset, *v)?),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic add on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Max => match &val {
                Value::U32(v) => Value::U32(self.shared.atomic_max_u32(handle_idx, offset, *v)?),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic max on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Min => match &val {
                Value::U32(v) => Value::U32(self.shared.atomic_min_u32(handle_idx, offset, *v)?),
                _ => {
                    return Err(NagaExecError::TypeMismatch(format!(
                        "atomic min on {val:?}"
                    )));
                }
            },
            naga::AtomicFunction::Exchange { compare: None } => {
                let old_val = Value::U32(self.shared.atomic_load_u32(handle_idx, offset)?);
                self.shared
                    .atomic_store_u32(handle_idx, offset, val.as_u32()?)?;
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
                let idx_u = idx.as_u32()? as usize;
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
            self.shared.write(gv_handle.index(), offset, &tmp)?;
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

#[cfg(test)]
#[path = "executor_tests.rs"]
mod tests;
