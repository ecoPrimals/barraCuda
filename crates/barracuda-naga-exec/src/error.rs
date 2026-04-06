// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for naga IR interpretation.

/// Errors during WGSL parsing, validation, or interpretation.
#[derive(Debug, thiserror::Error)]
pub enum NagaExecError {
    /// WGSL source failed to parse.
    #[error("WGSL parse error: {0}")]
    Parse(String),

    /// Parsed module failed naga validation.
    #[error("naga validation error: {0}")]
    Validation(String),

    /// Named entry point does not exist in the module.
    #[error("entry point '{0}' not found")]
    EntryPointNotFound(String),

    /// Entry point exists but is not a compute shader.
    #[error("entry point '{0}' is not a compute shader")]
    NotCompute(String),

    /// A `@group(g) @binding(b)` referenced by the shader is missing.
    #[error("binding ({group}, {binding}) not found in dispatch bindings")]
    BindingNotFound {
        /// Bind group index.
        group: u32,
        /// Binding index within the group.
        binding: u32,
    },

    /// Supplied buffer is smaller than the shader requires.
    #[error("buffer too small: need {need} bytes, have {have}")]
    BufferTooSmall {
        /// Required buffer size in bytes.
        need: usize,
        /// Actual buffer size in bytes.
        have: usize,
    },

    /// An expression node in the naga IR is not yet supported.
    #[error("unsupported naga expression: {0}")]
    UnsupportedExpression(String),

    /// A statement node in the naga IR is not yet supported.
    #[error("unsupported naga statement: {0}")]
    UnsupportedStatement(String),

    /// A type in the naga IR is not yet supported.
    #[error("unsupported naga type: {0}")]
    UnsupportedType(String),

    /// A `MathFunction` variant is not yet interpreted.
    #[error("unsupported math builtin: {0:?}")]
    UnsupportedMathBuiltin(naga::MathFunction),

    /// Value type did not match expected type during evaluation.
    #[error("type mismatch: {0}")]
    TypeMismatch(String),

    /// Array or composite access exceeded bounds.
    #[error("out of bounds access at index {index}, length {length}")]
    OutOfBounds {
        /// The index that was accessed.
        index: usize,
        /// The length of the container.
        length: usize,
    },
}

pub type Result<T> = std::result::Result<T, NagaExecError>;
