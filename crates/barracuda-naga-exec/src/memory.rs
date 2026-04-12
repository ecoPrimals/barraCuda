// SPDX-License-Identifier: AGPL-3.0-or-later
//! Memory operations for the naga IR CPU interpreter.
//!
//! Pointer load/store, buffer element access, and atomic operations.
//! Extracted from `invocation.rs` to keep the interpreter under 500 lines.

use naga::{AddressSpace, Expression, GlobalVariable, Handle, TypeInner};
use tracing::trace;

use crate::error::{NagaExecError, Result};
use crate::eval::type_byte_size;
use crate::invocation::InvocationContext;
use crate::value::Value;
use crate::vector_ops::access_index_val;

impl InvocationContext<'_> {
    pub(crate) fn load_pointer(&mut self, pointer: Handle<Expression>) -> Result<Value> {
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
                    let elements = (0..count)
                        .map(|i| Value::read_from_buffer(&buf.data, i * elem_size, elem_ty))
                        .collect();
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

    pub(crate) fn store_pointer(&mut self, pointer: Handle<Expression>, val: &Value) -> Result<()> {
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

    pub(crate) fn execute_atomic(
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

    pub(crate) fn load_buffer_element(
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

    pub(crate) fn store_buffer_element(
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
