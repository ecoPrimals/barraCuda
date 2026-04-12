// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-invocation execution context for naga IR interpretation.
//!
//! Each GPU thread maps to one `InvocationContext` that walks naga
//! statements and evaluates expressions against bound storage buffers,
//! workgroup shared memory, and per-invocation local variables.

use std::collections::BTreeMap;

use crate::error::{NagaExecError, Result};
use crate::eval::{compose_vector, eval_binary, eval_cast, eval_math, eval_unary};
use crate::sim_buffer::SimBuffer;
use crate::value::Value;
use crate::vector_ops::{access_index_val, splat_value, swizzle_value};
use crate::workgroup::{InvocationState, WorkgroupMemory};
use naga::{
    AddressSpace, Expression, Function, GlobalVariable, Handle, Literal, Module, Statement,
    TypeInner,
};

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
///
/// Memory operations (load/store/atomic/buffer) live in the sibling
/// [`memory`](super::memory) module to keep this file focused on
/// expression evaluation and statement dispatch.
pub(crate) struct InvocationContext<'a> {
    pub(crate) module: &'a Module,
    #[expect(
        dead_code,
        reason = "retained for future validation-aware interpretation"
    )]
    pub(crate) info: &'a naga::valid::ModuleInfo,
    pub(crate) function: &'a Function,
    pub(crate) bindings: &'a mut BTreeMap<(u32, u32), SimBuffer>,
    pub(crate) shared: &'a mut WorkgroupMemory,
    pub(crate) global_id: [u32; 3],
    pub(crate) local_id: [u32; 3],
    pub(crate) workgroup_id: [u32; 3],
    pub(crate) workgroup_size: [u32; 3],
    pub(crate) num_workgroups: [u32; 3],
    pub(crate) expr_cache: BTreeMap<Handle<Expression>, Value>,
    pub(crate) local_vars: BTreeMap<Handle<naga::LocalVariable>, Value>,
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

    pub(crate) fn get_value(&mut self, handle: Handle<Expression>) -> Result<Value> {
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

    pub(crate) fn resolve_binding(&self, gv_handle: Handle<GlobalVariable>) -> Option<(u32, u32)> {
        let gv = &self.module.global_variables[gv_handle];
        gv.binding.as_ref().map(|b| (b.group, b.binding))
    }
}
