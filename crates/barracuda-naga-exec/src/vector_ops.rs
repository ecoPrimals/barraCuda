// SPDX-License-Identifier: AGPL-3.0-or-later
//! Vector component access, splat, swizzle, and extraction helpers.
//!
//! These operate on [`Value`] enum variants and are used by the expression
//! evaluator in [`crate::executor`] for WGSL vector operations.

use crate::error::{NagaExecError, Result};
use crate::value::Value;

pub(crate) fn access_index_val(val: &Value, index: usize) -> Result<Value> {
    match val {
        Value::Vec2(v) => Ok(Value::F32(v[index])),
        Value::Vec3(v) => Ok(Value::F32(v[index])),
        Value::Vec4(v) => Ok(Value::F32(v[index])),
        Value::Vec2F64(v) => Ok(Value::F64(v[index])),
        Value::Vec3F64(v) => Ok(Value::F64(v[index])),
        Value::Vec4F64(v) => Ok(Value::F64(v[index])),
        Value::Vec2U32(v) => Ok(Value::U32(v[index])),
        Value::Vec3U32(v) => Ok(Value::U32(v[index])),
        Value::Vec4U32(v) => Ok(Value::U32(v[index])),
        Value::Vec2I32(v) => Ok(Value::I32(v[index])),
        Value::Vec3I32(v) => Ok(Value::I32(v[index])),
        Value::Vec4I32(v) => Ok(Value::I32(v[index])),
        Value::Composite(fields) => fields
            .get(index)
            .cloned()
            .ok_or(NagaExecError::OutOfBounds {
                index,
                length: fields.len(),
            }),
        _ => Err(NagaExecError::UnsupportedExpression(format!(
            "AccessIndex on {val:?}"
        ))),
    }
}

pub(crate) fn splat_value(size: naga::VectorSize, v: &Value) -> Result<Value> {
    let n = match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    };
    match v {
        Value::F32(f) => match n {
            2 => Ok(Value::Vec2([*f; 2])),
            3 => Ok(Value::Vec3([*f; 3])),
            _ => Ok(Value::Vec4([*f; 4])),
        },
        Value::F64(f) => match n {
            2 => Ok(Value::Vec2F64([*f; 2])),
            3 => Ok(Value::Vec3F64([*f; 3])),
            _ => Ok(Value::Vec4F64([*f; 4])),
        },
        Value::U32(u) => match n {
            2 => Ok(Value::Vec2U32([*u; 2])),
            3 => Ok(Value::Vec3U32([*u; 3])),
            _ => Ok(Value::Vec4U32([*u; 4])),
        },
        Value::I32(i) => match n {
            2 => Ok(Value::Vec2I32([*i; 2])),
            3 => Ok(Value::Vec3I32([*i; 3])),
            _ => Ok(Value::Vec4I32([*i; 4])),
        },
        _ => Err(NagaExecError::UnsupportedExpression(format!(
            "Splat of {v:?}"
        ))),
    }
}

fn swizzle_component_index(c: naga::SwizzleComponent) -> usize {
    match c {
        naga::SwizzleComponent::X => 0,
        naga::SwizzleComponent::Y => 1,
        naga::SwizzleComponent::Z => 2,
        naga::SwizzleComponent::W => 3,
    }
}

fn extract_f32_components(v: &Value) -> Option<&[f32]> {
    match v {
        Value::Vec2(a) => Some(a.as_slice()),
        Value::Vec3(a) => Some(a.as_slice()),
        Value::Vec4(a) => Some(a.as_slice()),
        _ => None,
    }
}

fn extract_f64_components(v: &Value) -> Option<&[f64]> {
    match v {
        Value::Vec2F64(a) => Some(a.as_slice()),
        Value::Vec3F64(a) => Some(a.as_slice()),
        Value::Vec4F64(a) => Some(a.as_slice()),
        _ => None,
    }
}

fn extract_u32_components(v: &Value) -> Option<&[u32]> {
    match v {
        Value::Vec2U32(a) => Some(a.as_slice()),
        Value::Vec3U32(a) => Some(a.as_slice()),
        Value::Vec4U32(a) => Some(a.as_slice()),
        _ => None,
    }
}

fn extract_i32_components(v: &Value) -> Option<&[i32]> {
    match v {
        Value::Vec2I32(a) => Some(a.as_slice()),
        Value::Vec3I32(a) => Some(a.as_slice()),
        Value::Vec4I32(a) => Some(a.as_slice()),
        _ => None,
    }
}

pub(crate) fn swizzle_value(
    size: naga::VectorSize,
    pattern: [naga::SwizzleComponent; 4],
    base: &Value,
) -> Result<Value> {
    let n = match size {
        naga::VectorSize::Bi => 2,
        naga::VectorSize::Tri => 3,
        naga::VectorSize::Quad => 4,
    };

    if let Some(comps) = extract_f32_components(base) {
        let mut out = [0.0f32; 4];
        for i in 0..n {
            out[i] = comps[swizzle_component_index(pattern[i])];
        }
        return Ok(match n {
            2 => Value::Vec2([out[0], out[1]]),
            3 => Value::Vec3([out[0], out[1], out[2]]),
            _ => Value::Vec4(out),
        });
    }
    if let Some(comps) = extract_f64_components(base) {
        let mut out = [0.0f64; 4];
        for i in 0..n {
            out[i] = comps[swizzle_component_index(pattern[i])];
        }
        return Ok(match n {
            2 => Value::Vec2F64([out[0], out[1]]),
            3 => Value::Vec3F64([out[0], out[1], out[2]]),
            _ => Value::Vec4F64(out),
        });
    }
    if let Some(comps) = extract_u32_components(base) {
        let mut out = [0u32; 4];
        for i in 0..n {
            out[i] = comps[swizzle_component_index(pattern[i])];
        }
        return Ok(match n {
            2 => Value::Vec2U32([out[0], out[1]]),
            3 => Value::Vec3U32([out[0], out[1], out[2]]),
            _ => Value::Vec4U32(out),
        });
    }
    if let Some(comps) = extract_i32_components(base) {
        let mut out = [0i32; 4];
        for i in 0..n {
            out[i] = comps[swizzle_component_index(pattern[i])];
        }
        return Ok(match n {
            2 => Value::Vec2I32([out[0], out[1]]),
            3 => Value::Vec3I32([out[0], out[1], out[2]]),
            _ => Value::Vec4I32(out),
        });
    }

    Err(NagaExecError::UnsupportedExpression(format!(
        "Swizzle on {base:?}"
    )))
}
