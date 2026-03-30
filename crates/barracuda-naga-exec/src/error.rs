// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for naga IR interpretation.

/// Errors during WGSL parsing, validation, or interpretation.
#[derive(Debug, thiserror::Error)]
pub enum NagaExecError {
    #[error("WGSL parse error: {0}")]
    Parse(String),

    #[error("naga validation error: {0}")]
    Validation(String),

    #[error("entry point '{0}' not found")]
    EntryPointNotFound(String),

    #[error("entry point '{0}' is not a compute shader")]
    NotCompute(String),

    #[error("binding ({group}, {binding}) not found in dispatch bindings")]
    BindingNotFound { group: u32, binding: u32 },

    #[error("buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    #[error("unsupported naga expression: {0}")]
    UnsupportedExpression(String),

    #[error("unsupported naga statement: {0}")]
    UnsupportedStatement(String),

    #[error("unsupported naga type: {0}")]
    UnsupportedType(String),

    #[error("unsupported math builtin: {0:?}")]
    UnsupportedMathBuiltin(naga::MathFunction),

    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    #[error("out of bounds access at index {index}, length {length}")]
    OutOfBounds { index: usize, length: usize },
}

pub type Result<T> = std::result::Result<T, NagaExecError>;
