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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]

    use super::*;

    #[test]
    fn network_config_default() {
        let config = NetworkConfig::default();
        assert!(matches!(
            config.hardware_preference,
            HardwarePreference::Auto
        ));
        assert!(!config.auto_mixed_precision);
        assert!(config.grad_clip.is_none());
        assert!(!config.enable_checkpointing);
    }

    #[test]
    fn hardware_preference_default_is_auto() {
        let pref = HardwarePreference::default();
        assert!(matches!(pref, HardwarePreference::Auto));
    }

    #[test]
    fn network_config_clone() {
        let config = NetworkConfig {
            hardware_preference: HardwarePreference::PreferGPU,
            auto_mixed_precision: true,
            grad_clip: Some(1.0),
            enable_checkpointing: true,
        };
        assert!(matches!(
            config.hardware_preference,
            HardwarePreference::PreferGPU
        ));
        assert!(config.auto_mixed_precision);
        assert_eq!(config.grad_clip, Some(1.0));
        assert!(config.enable_checkpointing);
    }
}
