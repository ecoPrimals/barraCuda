// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime value representation for the naga IR interpreter.

use crate::error::{NagaExecError, Result};

/// A runtime value during interpretation.
///
/// Mirrors naga's type system but holds concrete data. All numeric types
/// are represented natively — f64 is NOT emulated, giving us precision
/// that llvmpipe cannot provide.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
    F64(f64),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Vec2F64([f64; 2]),
    Vec3F64([f64; 3]),
    Vec4F64([f64; 4]),
    Vec2U32([u32; 2]),
    Vec3U32([u32; 3]),
    Vec4U32([u32; 4]),
    Vec2I32([i32; 2]),
    Vec3I32([i32; 3]),
    Vec4I32([i32; 4]),
    /// Composite (struct or array) — indexed by position.
    Composite(Vec<Value>),
}

impl Value {
    /// Extract as f32, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::TypeMismatch`] if `self` is not a numeric scalar.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "WGSL numeric coercions: naga type system guarantees valid source types"
    )]
    pub fn as_f32(&self) -> Result<f32> {
        match self {
            Self::F32(v) => Ok(*v),
            Self::F64(v) => Ok(*v as f32),
            Self::I32(v) => Ok(*v as f32),
            Self::U32(v) => Ok(*v as f32),
            _ => Err(NagaExecError::TypeMismatch(format!(
                "expected f32-coercible, got {self:?}"
            ))),
        }
    }

    /// Extract as f64, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::TypeMismatch`] if `self` is not a numeric scalar.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            Self::F64(v) => Ok(*v),
            Self::F32(v) => Ok(f64::from(*v)),
            Self::I32(v) => Ok(f64::from(*v)),
            Self::U32(v) => Ok(f64::from(*v)),
            _ => Err(NagaExecError::TypeMismatch(format!(
                "expected f64-coercible, got {self:?}"
            ))),
        }
    }

    /// Extract as u32, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::TypeMismatch`] if `self` is not a u32-coercible scalar.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "WGSL numeric coercions: naga type system guarantees valid source types"
    )]
    pub fn as_u32(&self) -> Result<u32> {
        match self {
            Self::U32(v) => Ok(*v),
            Self::I32(v) => Ok(*v as u32),
            Self::F32(v) => Ok(*v as u32),
            Self::Bool(v) => Ok(u32::from(*v)),
            _ => Err(NagaExecError::TypeMismatch(format!(
                "expected u32-coercible, got {self:?}"
            ))),
        }
    }

    /// Extract as i32, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::TypeMismatch`] if `self` is not an i32-coercible scalar.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "WGSL numeric coercions: f32 as i32 may truncate"
    )]
    pub fn as_i32(&self) -> Result<i32> {
        match self {
            Self::I32(v) => Ok(*v),
            Self::U32(v) => Ok(v.cast_signed()),
            Self::F32(v) => Ok(*v as i32),
            _ => Err(NagaExecError::TypeMismatch(format!(
                "expected i32-coercible, got {self:?}"
            ))),
        }
    }

    /// Extract as bool, returning an error on type mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`NagaExecError::TypeMismatch`] if `self` is not a bool-coercible scalar.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            Self::Bool(v) => Ok(*v),
            Self::U32(v) => Ok(*v != 0),
            Self::I32(v) => Ok(*v != 0),
            _ => Err(NagaExecError::TypeMismatch(format!(
                "expected bool-coercible, got {self:?}"
            ))),
        }
    }

    /// Read this value from a byte buffer at the given offset.
    #[must_use]
    pub fn read_from_buffer(buf: &[u8], offset: usize, ty: &naga::Type) -> Self {
        match ty.inner {
            naga::TypeInner::Scalar(s) | naga::TypeInner::Atomic(s) => {
                Self::read_scalar(buf, offset, s)
            }
            naga::TypeInner::Vector { size, scalar } => {
                Self::read_vector(buf, offset, size, scalar)
            }
            _ => Self::U32(0),
        }
    }

    /// Write this value into a byte buffer at the given offset.
    pub fn write_to_buffer(&self, buf: &mut [u8], offset: usize) {
        match self {
            Self::F32(v) => buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes()),
            Self::F64(v) => buf[offset..offset + 8].copy_from_slice(&v.to_le_bytes()),
            Self::U32(v) => buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes()),
            Self::I32(v) => buf[offset..offset + 4].copy_from_slice(&v.to_le_bytes()),
            Self::Bool(v) => buf[offset..offset + 4].copy_from_slice(&u32::from(*v).to_le_bytes()),
            Self::Vec2(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec3(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec4(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec2F64(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 8..offset + (i + 1) * 8].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec3F64(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 8..offset + (i + 1) * 8].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec4F64(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 8..offset + (i + 1) * 8].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec2U32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec3U32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec4U32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec2I32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec3I32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Vec4I32(v) => {
                for (i, c) in v.iter().enumerate() {
                    buf[offset + i * 4..offset + (i + 1) * 4].copy_from_slice(&c.to_le_bytes());
                }
            }
            Self::Composite(_) => {}
        }
    }

    fn read_scalar(buf: &[u8], offset: usize, scalar: naga::Scalar) -> Self {
        match (scalar.kind, scalar.width) {
            (naga::ScalarKind::Float, 4) => Self::F32(f32::from_le_bytes(
                buf[offset..offset + 4].try_into().unwrap_or([0; 4]),
            )),
            (naga::ScalarKind::Float, 8) => Self::F64(f64::from_le_bytes(
                buf[offset..offset + 8].try_into().unwrap_or([0; 8]),
            )),
            (naga::ScalarKind::Uint, 4) => Self::U32(u32::from_le_bytes(
                buf[offset..offset + 4].try_into().unwrap_or([0; 4]),
            )),
            (naga::ScalarKind::Sint, 4) => Self::I32(i32::from_le_bytes(
                buf[offset..offset + 4].try_into().unwrap_or([0; 4]),
            )),
            (naga::ScalarKind::Bool, _) => Self::Bool(
                u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4])) != 0,
            ),
            _ => Self::U32(0),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "exhaustive match over all vector type/size combinations"
    )]
    fn read_vector(
        buf: &[u8],
        offset: usize,
        size: naga::VectorSize,
        scalar: naga::Scalar,
    ) -> Self {
        let n = match size {
            naga::VectorSize::Bi => 2,
            naga::VectorSize::Tri => 3,
            naga::VectorSize::Quad => 4,
        };
        let w = scalar.width as usize;
        match (scalar.kind, scalar.width, n) {
            (naga::ScalarKind::Float, 4, 2) => {
                let mut v = [0f32; 2];
                for i in 0..2 {
                    v[i] = f32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec2(v)
            }
            (naga::ScalarKind::Float, 4, 3) => {
                let mut v = [0f32; 3];
                for i in 0..3 {
                    v[i] = f32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec3(v)
            }
            (naga::ScalarKind::Float, 4, 4) => {
                let mut v = [0f32; 4];
                for i in 0..4 {
                    v[i] = f32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec4(v)
            }
            (naga::ScalarKind::Float, 8, 2) => {
                let mut v = [0f64; 2];
                for i in 0..2 {
                    v[i] = f64::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 8]),
                    );
                }
                Self::Vec2F64(v)
            }
            (naga::ScalarKind::Float, 8, 3) => {
                let mut v = [0f64; 3];
                for i in 0..3 {
                    v[i] = f64::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 8]),
                    );
                }
                Self::Vec3F64(v)
            }
            (naga::ScalarKind::Float, 8, 4) => {
                let mut v = [0f64; 4];
                for i in 0..4 {
                    v[i] = f64::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 8]),
                    );
                }
                Self::Vec4F64(v)
            }
            (naga::ScalarKind::Uint, 4, 2) => {
                let mut v = [0u32; 2];
                for i in 0..2 {
                    v[i] = u32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec2U32(v)
            }
            (naga::ScalarKind::Uint, 4, 3) => {
                let mut v = [0u32; 3];
                for i in 0..3 {
                    v[i] = u32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec3U32(v)
            }
            (naga::ScalarKind::Uint, 4, 4) => {
                let mut v = [0u32; 4];
                for i in 0..4 {
                    v[i] = u32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec4U32(v)
            }
            (naga::ScalarKind::Sint, 4, 2) => {
                let mut v = [0i32; 2];
                for i in 0..2 {
                    v[i] = i32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec2I32(v)
            }
            (naga::ScalarKind::Sint, 4, 3) => {
                let mut v = [0i32; 3];
                for i in 0..3 {
                    v[i] = i32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec3I32(v)
            }
            (naga::ScalarKind::Sint, 4, 4) => {
                let mut v = [0i32; 4];
                for i in 0..4 {
                    v[i] = i32::from_le_bytes(
                        buf[offset + i * w..offset + (i + 1) * w]
                            .try_into()
                            .unwrap_or([0; 4]),
                    );
                }
                Self::Vec4I32(v)
            }
            _ => Self::U32(0),
        }
    }

    /// Size of this value in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Bool(_) | Self::I32(_) | Self::U32(_) | Self::F32(_) => 4,
            Self::F64(_) | Self::Vec2(_) | Self::Vec2I32(_) | Self::Vec2U32(_) => 8,
            Self::Vec3(_) | Self::Vec3I32(_) | Self::Vec3U32(_) => 12,
            Self::Vec4(_) | Self::Vec4I32(_) | Self::Vec4U32(_) | Self::Vec2F64(_) => 16,
            Self::Vec3F64(_) => 24,
            Self::Vec4F64(_) => 32,
            Self::Composite(v) => v.iter().map(Self::byte_size).sum(),
        }
    }
}
