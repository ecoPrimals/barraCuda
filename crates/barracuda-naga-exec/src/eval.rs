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

#[expect(
    clippy::too_many_lines,
    reason = "exhaustive match over MathFunction variants for each scalar type"
)]
pub(crate) fn eval_math(fun: MathFunction, a: &Value, b: Option<&Value>) -> Result<Value> {
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
                        .as_f32()?;
                    x.powf(y)
                }
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_f32()?;
                    x.min(y)
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_f32()?;
                    x.max(y)
                }
                MathFunction::Atan2 => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("atan2 needs 2 args".into()))?
                        .as_f32()?;
                    x.atan2(y)
                }
                MathFunction::Step => {
                    let edge = b
                        .ok_or(NagaExecError::TypeMismatch("step needs 2 args".into()))?
                        .as_f32()?;
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
                        .as_f64()?;
                    x.powf(y)
                }
                MathFunction::Min => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("min needs 2 args".into()))?
                        .as_f64()?;
                    x.min(y)
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_f64()?;
                    x.max(y)
                }
                MathFunction::Atan2 => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("atan2 needs 2 args".into()))?
                        .as_f64()?;
                    x.atan2(y)
                }
                MathFunction::Step => {
                    let edge = b
                        .ok_or(NagaExecError::TypeMismatch("step needs 2 args".into()))?
                        .as_f64()?;
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
                        .as_u32()?;
                    Value::U32(x.min(y))
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_u32()?;
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
                        .as_i32()?;
                    Value::I32(x.min(y))
                }
                MathFunction::Max => {
                    let y = b
                        .ok_or(NagaExecError::TypeMismatch("max needs 2 args".into()))?
                        .as_i32()?;
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
        (ScalarKind::Uint, 4, naga::VectorSize::Tri) => Ok(Value::Vec3U32([
            vals[0].as_u32()?,
            vals[1].as_u32()?,
            vals[2].as_u32()?,
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
