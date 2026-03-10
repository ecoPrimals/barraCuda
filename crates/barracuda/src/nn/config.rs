// SPDX-License-Identifier: AGPL-3.0-only
//! Neural Network Configuration
//!
//! Runtime configuration for neural network training and inference.
//! Deep Debt compliant: No hardcoded values, all runtime configurable.

/// Network configuration (runtime, no hardcoding)
#[derive(Debug, Clone, Default)]
pub struct NetworkConfig {
    /// Hardware preference (discovered at runtime)
    pub hardware_preference: HardwarePreference,

    /// Enable automatic mixed precision
    pub auto_mixed_precision: bool,

    /// Gradient clipping threshold
    pub grad_clip: Option<f32>,

    /// Enable checkpointing
    pub enable_checkpointing: bool,
}

/// Hardware preference (runtime discovery)
#[derive(Debug, Clone, Default)]
pub enum HardwarePreference {
    /// Automatic selection (recommended)
    #[default]
    Auto,
    /// Prefer GPU if available
    PreferGPU,
    /// Prefer NPU if available
    PreferNPU,
    /// CPU only
    CPUOnly,
}
