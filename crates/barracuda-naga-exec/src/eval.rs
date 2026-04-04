// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pure evaluation helpers for naga IR arithmetic, math builtins, casts,
//! vector composition, and type sizing.

use naga::{BinaryOperator, MathFunction, Module, ScalarKind, TypeInner, UnaryOperator};

use crate::error::{NagaExecError, Result};
use crate::value::Value;

pub(crate) fn eval_binary(op: BinaryOperator, l: &Value, r: &Value) -> Result<Value> {
    match (l, r) {
        (Value::F32(a), Value::F32(b)) => binary_f32(op, *a, *b),
        (Value::F64(a), Value::F64(b)) => binary_f64(op, *a, *b),
        (Value::U32(a), Value::U32(b)) => binary_u32(op, *a, *b),
        (Value::I32(a), Value::I32(b)) => binary_i32(op, *a, *b),
        (Value::Bool(a), Value::Bool(b)) => binary_bool(op, *a, *b),
        (Value::Vec2(a), Value::Vec2(b)) => binary_f32_arr::<2>(op, a, b).map(Value::Vec2),
        (Value::Vec3(a), Value::Vec3(b)) => binary_f32_arr::<3>(op, a, b).map(Value::Vec3),
        (Value::Vec4(a), Value::Vec4(b)) => binary_f32_arr::<4>(op, a, b).map(Value::Vec4),
        (Value::Vec2F64(a), Value::Vec2F64(b)) => binary_f64_arr::<2>(op, a, b).map(Value::Vec2F64),
        (Value::Vec3F64(a), Value::Vec3F64(b)) => binary_f64_arr::<3>(op, a, b).map(Value::Vec3F64),
        (Value::Vec4F64(a), Value::Vec4F64(b)) => binary_f64_arr::<4>(op, a, b).map(Value::Vec4F64),
        (Value::Vec2U32(a), Value::Vec2U32(b)) => binary_u32_arr::<2>(op, a, b).map(Value::Vec2U32),
        (Value::Vec3U32(a), Value::Vec3U32(b)) => binary_u32_arr::<3>(op, a, b).map(Value::Vec3U32),
        (Value::Vec4U32(a), Value::Vec4U32(b)) => binary_u32_arr::<4>(op, a, b).map(Value::Vec4U32),
        (Value::Vec2I32(a), Value::Vec2I32(b)) => binary_i32_arr::<2>(op, a, b).map(Value::Vec2I32),
        (Value::Vec3I32(a), Value::Vec3I32(b)) => binary_i32_arr::<3>(op, a, b).map(Value::Vec3I32),
        (Value::Vec4I32(a), Value::Vec4I32(b)) => binary_i32_arr::<4>(op, a, b).map(Value::Vec4I32),
        _ => Err(NagaExecError::TypeMismatch(format!(
            "binary {op:?} on {l:?} and {r:?}"
        ))),
    }
}

fn binary_f32_arr<const N: usize>(
    op: BinaryOperator,
    a: &[f32; N],
    b: &[f32; N],
) -> Result<[f32; N]> {
    let mut out = [0.0f32; N];
    for i in 0..N {
        out[i] = match binary_f32(op, a[i], b[i])? {
            Value::F32(v) => v,
            Value::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
    }
    Ok(out)
}

fn binary_f64_arr<const N: usize>(
    op: BinaryOperator,
    a: &[f64; N],
    b: &[f64; N],
) -> Result<[f64; N]> {
    let mut out = [0.0f64; N];
    for i in 0..N {
        out[i] = match binary_f64(op, a[i], b[i])? {
            Value::F64(v) => v,
            Value::Bool(v) => {
                if v {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
    }
    Ok(out)
}

fn binary_u32_arr<const N: usize>(
    op: BinaryOperator,
    a: &[u32; N],
    b: &[u32; N],
) -> Result<[u32; N]> {
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = match binary_u32(op, a[i], b[i])? {
            Value::U32(v) => v,
            Value::Bool(v) => u32::from(v),
            _ => 0,
        };
    }
    Ok(out)
}

fn binary_i32_arr<const N: usize>(
    op: BinaryOperator,
    a: &[i32; N],
    b: &[i32; N],
) -> Result<[i32; N]> {
    let mut out = [0i32; N];
    for i in 0..N {
        out[i] = match binary_i32(op, a[i], b[i])? {
            Value::I32(v) => v,
            Value::Bool(v) => i32::from(v),
            _ => 0,
        };
    }
    Ok(out)
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

pub(crate) fn eval_unary(op: UnaryOperator, v: &Value) -> Result<Value> {
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

pub(crate) fn eval_math(
    fun: MathFunction,
    a: &Value,
    b: Option<&Value>,
    c: Option<&Value>,
) -> Result<Value> {
    match a {
        Value::F32(x) => math_f32(fun, *x, b, c).map(Value::F32),
        Value::F64(x) => math_f64(fun, *x, b, c).map(Value::F64),
        Value::U32(x) => math_u32(fun, *x, b),
        Value::I32(x) => math_i32(fun, *x, b),
        _ => Err(NagaExecError::UnsupportedExpression(format!(
            "Math({fun:?}) on {a:?}"
        ))),
    }
}

#[expect(clippy::many_single_char_names, reason = "standard math notation")]
fn math_f32(fun: MathFunction, x: f32, b: Option<&Value>, c: Option<&Value>) -> Result<f32> {
    Ok(match fun {
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
        MathFunction::Pow => x.powf(require_arg(b, "pow")?.as_f32()?),
        MathFunction::Min => x.min(require_arg(b, "min")?.as_f32()?),
        MathFunction::Max => x.max(require_arg(b, "max")?.as_f32()?),
        MathFunction::Atan2 => x.atan2(require_arg(b, "atan2")?.as_f32()?),
        MathFunction::Step => {
            let edge = require_arg(b, "step")?.as_f32()?;
            if x >= edge { 1.0 } else { 0.0 }
        }
        MathFunction::Clamp => x.clamp(
            require_arg(b, "clamp")?.as_f32()?,
            require_arg(c, "clamp")?.as_f32()?,
        ),
        MathFunction::Mix => {
            let y = require_arg(b, "mix")?.as_f32()?;
            let t = require_arg(c, "mix")?.as_f32()?;
            x.mul_add(1.0 - t, y * t)
        }
        MathFunction::SmoothStep => {
            let high = require_arg(b, "smoothstep")?.as_f32()?;
            let val = require_arg(c, "smoothstep")?.as_f32()?;
            let t = ((val - x) / (high - x)).clamp(0.0, 1.0);
            t * t * 2.0f32.mul_add(-t, 3.0)
        }
        MathFunction::Fma => x.mul_add(
            require_arg(b, "fma")?.as_f32()?,
            require_arg(c, "fma")?.as_f32()?,
        ),
        _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
    })
}

#[expect(clippy::many_single_char_names, reason = "standard math notation")]
fn math_f64(fun: MathFunction, x: f64, b: Option<&Value>, c: Option<&Value>) -> Result<f64> {
    Ok(match fun {
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
        MathFunction::Pow => x.powf(require_arg(b, "pow")?.as_f64()?),
        MathFunction::Min => x.min(require_arg(b, "min")?.as_f64()?),
        MathFunction::Max => x.max(require_arg(b, "max")?.as_f64()?),
        MathFunction::Atan2 => x.atan2(require_arg(b, "atan2")?.as_f64()?),
        MathFunction::Step => {
            let edge = require_arg(b, "step")?.as_f64()?;
            if x >= edge { 1.0 } else { 0.0 }
        }
        MathFunction::Clamp => x.clamp(
            require_arg(b, "clamp")?.as_f64()?,
            require_arg(c, "clamp")?.as_f64()?,
        ),
        MathFunction::Mix => {
            let y = require_arg(b, "mix")?.as_f64()?;
            let t = require_arg(c, "mix")?.as_f64()?;
            x.mul_add(1.0 - t, y * t)
        }
        MathFunction::SmoothStep => {
            let high = require_arg(b, "smoothstep")?.as_f64()?;
            let val = require_arg(c, "smoothstep")?.as_f64()?;
            let t = ((val - x) / (high - x)).clamp(0.0, 1.0);
            t * t * 2.0f64.mul_add(-t, 3.0)
        }
        MathFunction::Fma => x.mul_add(
            require_arg(b, "fma")?.as_f64()?,
            require_arg(c, "fma")?.as_f64()?,
        ),
        _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
    })
}

fn math_u32(fun: MathFunction, x: u32, b: Option<&Value>) -> Result<Value> {
    Ok(match fun {
        MathFunction::Min => Value::U32(x.min(require_arg(b, "min")?.as_u32()?)),
        MathFunction::Max => Value::U32(x.max(require_arg(b, "max")?.as_u32()?)),
        _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
    })
}

fn math_i32(fun: MathFunction, x: i32, b: Option<&Value>) -> Result<Value> {
    Ok(match fun {
        MathFunction::Abs => Value::I32(x.abs()),
        MathFunction::Min => Value::I32(x.min(require_arg(b, "min")?.as_i32()?)),
        MathFunction::Max => Value::I32(x.max(require_arg(b, "max")?.as_i32()?)),
        _ => return Err(NagaExecError::UnsupportedMathBuiltin(fun)),
    })
}

fn require_arg<'a>(arg: Option<&'a Value>, name: &str) -> Result<&'a Value> {
    arg.ok_or_else(|| NagaExecError::TypeMismatch(format!("{name} needs more arguments")))
}

pub(crate) fn eval_cast(v: &Value, kind: ScalarKind, convert: Option<u8>) -> Result<Value> {
    let width = convert.unwrap_or(4);
    Ok(match (kind, width) {
        (ScalarKind::Float, 4) => Value::F32(v.as_f32()?),
        (ScalarKind::Float, 8) => Value::F64(v.as_f64()?),
        (ScalarKind::Uint, 4) => Value::U32(v.as_u32()?),
        (ScalarKind::Sint, 4) => Value::I32(v.as_i32()?),
        (ScalarKind::Bool, _) => Value::Bool(v.as_bool()?),
        _ => {
            return Err(NagaExecError::UnsupportedType(format!(
                "cast to {kind:?} width={width}"
            )));
        }
    })
}

pub(crate) fn compose_vector(
    size: naga::VectorSize,
    scalar: naga::Scalar,
    vals: &[Value],
) -> Result<Value> {
    match (scalar.kind, scalar.width, size) {
        (ScalarKind::Float, 4, naga::VectorSize::Bi) => {
            Ok(Value::Vec2([vals[0].as_f32()?, vals[1].as_f32()?]))
        }
        (ScalarKind::Float, 4, naga::VectorSize::Tri) => Ok(Value::Vec3([
            vals[0].as_f32()?,
            vals[1].as_f32()?,
            vals[2].as_f32()?,
        ])),
        (ScalarKind::Float, 4, naga::VectorSize::Quad) => Ok(Value::Vec4([
            vals[0].as_f32()?,
            vals[1].as_f32()?,
            vals[2].as_f32()?,
            vals[3].as_f32()?,
        ])),
        (ScalarKind::Float, 8, naga::VectorSize::Bi) => {
            Ok(Value::Vec2F64([vals[0].as_f64()?, vals[1].as_f64()?]))
        }
        (ScalarKind::Float, 8, naga::VectorSize::Tri) => Ok(Value::Vec3F64([
            vals[0].as_f64()?,
            vals[1].as_f64()?,
            vals[2].as_f64()?,
        ])),
        (ScalarKind::Float, 8, naga::VectorSize::Quad) => Ok(Value::Vec4F64([
            vals[0].as_f64()?,
            vals[1].as_f64()?,
            vals[2].as_f64()?,
            vals[3].as_f64()?,
        ])),
        (ScalarKind::Uint, 4, naga::VectorSize::Bi) => {
            Ok(Value::Vec2U32([vals[0].as_u32()?, vals[1].as_u32()?]))
        }
        (ScalarKind::Uint, 4, naga::VectorSize::Tri) => Ok(Value::Vec3U32([
            vals[0].as_u32()?,
            vals[1].as_u32()?,
            vals[2].as_u32()?,
        ])),
        (ScalarKind::Uint, 4, naga::VectorSize::Quad) => Ok(Value::Vec4U32([
            vals[0].as_u32()?,
            vals[1].as_u32()?,
            vals[2].as_u32()?,
            vals[3].as_u32()?,
        ])),
        (ScalarKind::Sint, 4, naga::VectorSize::Bi) => {
            Ok(Value::Vec2I32([vals[0].as_i32()?, vals[1].as_i32()?]))
        }
        (ScalarKind::Sint, 4, naga::VectorSize::Tri) => Ok(Value::Vec3I32([
            vals[0].as_i32()?,
            vals[1].as_i32()?,
            vals[2].as_i32()?,
        ])),
        (ScalarKind::Sint, 4, naga::VectorSize::Quad) => Ok(Value::Vec4I32([
            vals[0].as_i32()?,
            vals[1].as_i32()?,
            vals[2].as_i32()?,
            vals[3].as_i32()?,
        ])),
        _ => Err(NagaExecError::UnsupportedType(format!(
            "compose vector {scalar:?} x {size:?}"
        ))),
    }
}

pub(crate) fn type_byte_size(module: &Module, inner: &TypeInner) -> usize {
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
