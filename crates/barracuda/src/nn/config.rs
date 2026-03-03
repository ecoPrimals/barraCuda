// SPDX-License-Identifier: AGPL-3.0-or-later
//! Neural Network Configuration
//!
//! Runtime configuration for neural network training and inference.
//! Deep Debt compliant: No hardcoded values, all runtime configurable.

/// Network configuration (runtime, no hardcoding)
#[derive(Debug, Clone)]
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

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            hardware_preference: HardwarePreference::Auto,
            auto_mixed_precision: false,
            grad_clip: None,
            enable_checkpointing: false,
        }
    }
}

/// Hardware preference (runtime discovery)
#[derive(Debug, Clone)]
pub enum HardwarePreference {
    /// Automatic selection (recommended)
    Auto,
    /// Prefer GPU if available
    PreferGPU,
    /// Prefer NPU if available
    PreferNPU,
    /// CPU only
    CPUOnly,
}
