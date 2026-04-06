// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-invocation execution context for naga IR interpretation.
//!
//! Each GPU thread maps to one `InvocationContext` that walks naga
//! statements and evaluates expressions against bound storage buffers,
//! workgroup shared memory, and per-invocation local variables.

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
use crate::vector_ops::{access_index_val, splat_value, swizzle_value};
use crate::workgroup::{InvocationState, WorkgroupMemory};

/// Maximum iterations for WGSL `loop` before aborting.
///
/// WGSL has no guaranteed termination — this bound prevents the CPU
/// interpreter from spinning indefinitely on pathological shaders.
const LOOP_ITERATION_LIMIT: usize = 100_000;

/// Control flow signal for structured WGSL execution.
pub(crate) enum Flow {
    Proceed,
    LoopBreak,
    LoopContinue,
    EarlyReturn,
}

/// GPU dispatch dimensions for a single invocation.
///
/// Bundles the five `[u32; 3]` fields that describe where a thread sits
/// in the dispatch grid, replacing a 10-parameter constructor.
#[derive(Clone, Copy)]
pub(crate) struct DispatchCoords {
    pub global_id: [u32; 3],
    pub local_id: [u32; 3],
    pub workgroup_id: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub num_workgroups: [u32; 3],
}

/// Per-invocation state: expression cache, local variables, buffer access.
pub(crate) struct InvocationContext<'a> {
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
    workgroup_id: [u32; 3],
    workgroup_size: [u32; 3],
    num_workgroups: [u32; 3],
    expr_cache: BTreeMap<Handle<Expression>, Value>,
    local_vars: BTreeMap<Handle<naga::LocalVariable>, Value>,
}

impl<'a> InvocationContext<'a> {
    pub(crate) fn new(
        module: &'a Module,
        info: &'a naga::valid::ModuleInfo,
        function: &'a Function,
        bindings: &'a mut BTreeMap<(u32, u32), SimBuffer>,
        shared: &'a mut WorkgroupMemory,
        coords: DispatchCoords,
    ) -> Self {
        Self {
            module,
            info,
            function,
            bindings,
            shared,
            global_id: coords.global_id,
            local_id: coords.local_id,
            workgroup_id: coords.workgroup_id,
            workgroup_size: coords.workgroup_size,
            num_workgroups: coords.num_workgroups,
            expr_cache: BTreeMap::new(),
            local_vars: BTreeMap::new(),
        }
    }

    pub(crate) fn save_state(&self) -> InvocationState {
        InvocationState {
            expr_cache: self.expr_cache.clone(),
            local_vars: self.local_vars.clone(),
        }
    }

    pub(crate) fn restore_state(&mut self, state: &InvocationState) {
        self.expr_cache.clone_from(&state.expr_cache);
        self.local_vars.clone_from(&state.local_vars);
    }

    pub(crate) fn execute_body(&mut self, body: &naga::Block) -> Result<Flow> {
        for stmt in body {
            match self.execute_stmt(stmt)? {
                Flow::Proceed => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Proceed)
    }

    fn execute_stmt(&mut self, stmt: &Statement) -> Result<Flow> {
        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let val = self.eval_expr(handle)?;
                    self.expr_cache.insert(handle, val);
                }
                Ok(Flow::Proceed)
            }
            Statement::Store { pointer, value } => {
                let val = self.get_value(value)?;
                self.store_pointer(pointer, &val)?;
                Ok(Flow::Proceed)
            }
            Statement::Block(ref block) => self.execute_body(block),
            Statement::Return { .. } => Ok(Flow::EarlyReturn),
            Statement::Break => Ok(Flow::LoopBreak),
            Statement::Continue => Ok(Flow::LoopContinue),
            Statement::ControlBarrier(_) | Statement::MemoryBarrier(_) => Ok(Flow::Proceed),
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                self.execute_atomic(pointer, fun, value, result)?;
                Ok(Flow::Proceed)
            }
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
                for _ in 0..LOOP_ITERATION_LIMIT {
                    match self.execute_body(body)? {
                        Flow::LoopBreak => break,
                        Flow::EarlyReturn => return Ok(Flow::EarlyReturn),
                        Flow::LoopContinue | Flow::Proceed => {}
                    }
                    match self.execute_body(continuing)? {
                        Flow::LoopBreak => break,
                        Flow::EarlyReturn => return Ok(Flow::EarlyReturn),
                        Flow::LoopContinue | Flow::Proceed => {}
                    }
                    if let Some(break_cond) = break_if {
                        let cond = self.get_value(break_cond)?;
                        if cond.as_bool()? {
                            break;
                        }
                    }
                }
                Ok(Flow::Proceed)
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                let sel = self.get_value(selector)?;
                let sel_i32 = sel.as_i32().unwrap_or(0);
                let mut matched = false;
                for case in cases {
                    let hit = match case.value {
                        naga::SwitchValue::I32(v) => v == sel_i32,
                        naga::SwitchValue::U32(v) => v == sel.as_u32().unwrap_or(0),
                        naga::SwitchValue::Default => !matched,
                    };
                    if hit || matched {
                        matched = true;
                        match self.execute_body(&case.body)? {
                            Flow::LoopBreak | Flow::Proceed => {
                                if case.fall_through {
                                    continue;
                                }
                                return Ok(Flow::Proceed);
                            }
                            Flow::EarlyReturn => return Ok(Flow::EarlyReturn),
                            Flow::LoopContinue => return Ok(Flow::LoopContinue),
                        }
                    }
                }
                Ok(Flow::Proceed)
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
                        Ok(Value::Vec3U32(self.num_workgroups))
                    }
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::WorkGroupId)) => {
                        Ok(Value::Vec3U32(self.workgroup_id))
                    }
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::WorkGroupSize)) => {
                        Ok(Value::Vec3U32(self.workgroup_size))
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

            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                ..
            } => {
                let a = self.get_value(arg)?;
                let b = arg1.map(|h| self.get_value(h)).transpose()?;
                let c = arg2.map(|h| self.get_value(h)).transpose()?;
                eval_math(fun, &a, b.as_ref(), c.as_ref())
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
                access_index_val(&base_val, index as usize)
            }

            Expression::Access { base, index } => {
                let idx = self.get_value(index)?;
                let idx_u = idx.as_u32()? as usize;

                let base_expr = &self.function.expressions[base];
                if let Expression::GlobalVariable(gv_handle) = *base_expr {
                    return self.load_buffer_element(gv_handle, idx_u);
                }

                let base_val = self.get_value(base)?;
                access_index_val(&base_val, idx_u)
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
                splat_value(size, &v)
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

            Expression::ArrayLength(expr_handle) => {
                let inner_expr = &self.function.expressions[expr_handle];
                if let Expression::GlobalVariable(gv_handle) = *inner_expr {
                    let gv = &self.module.global_variables[gv_handle];
                    let ty = &self.module.types[gv.ty];
                    if let TypeInner::Array { base, .. } = ty.inner {
                        let elem_size = crate::eval::type_byte_size(
                            self.module,
                            &self.module.types[base].inner,
                        );
                        let (group, binding) =
                            self.resolve_binding(gv_handle).ok_or_else(|| {
                                NagaExecError::UnsupportedExpression(
                                    "ArrayLength on variable without binding".into(),
                                )
                            })?;
                        let buf = self
                            .bindings
                            .get(&(group, binding))
                            .ok_or(NagaExecError::BindingNotFound { group, binding })?;
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "buffer element count fits u32 for WGSL arrays"
                        )]
                        return Ok(Value::U32((buf.data.len() / elem_size) as u32));
                    }
                }
                Err(NagaExecError::UnsupportedExpression(
                    "ArrayLength on non-array".into(),
                ))
            }

            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let base = self.get_value(vector)?;
                swizzle_value(size, pattern, &base)
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
                    let (group, binding) = self.resolve_binding(gv_handle).ok_or_else(|| {
                        NagaExecError::UnsupportedExpression(
                            "GlobalVariable without binding".into(),
                        )
                    })?;
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
                    let (group, binding) = self.resolve_binding(gv_handle).ok_or_else(|| {
                        NagaExecError::UnsupportedExpression(
                            "GlobalVariable without binding".into(),
                        )
                    })?;
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
                access_index_val(&base_val, idx_u)
            }
            Expression::AccessIndex { base, index } => {
                let base_val = self.load_pointer(base)?;
                access_index_val(&base_val, index as usize)
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
                let (group, binding) = self.resolve_binding(gv_handle).ok_or_else(|| {
                    NagaExecError::UnsupportedStatement(
                        "Store to GlobalVariable without binding".into(),
                    )
                })?;
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

        let (group, binding) = self.resolve_binding(gv_handle).ok_or_else(|| {
            NagaExecError::UnsupportedExpression("GlobalVariable without binding".into())
        })?;
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

        let (group, binding) = self.resolve_binding(gv_handle).ok_or_else(|| {
            NagaExecError::UnsupportedStatement("GlobalVariable without binding".into())
        })?;
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
