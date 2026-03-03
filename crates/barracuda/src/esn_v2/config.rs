// SPDX-License-Identifier: AGPL-3.0-or-later
//! ESN configuration, validation, and builder

use crate::error::{BarracudaError, Result as BarracudaResult};

// ── Multi-head ESN constants (hotSpring v0.6.15 absorption) ─────────────────

/// Number of output heads for multi-target ESN (hotSpring 11-head standard).
pub const ESN_MULTI_HEAD_COUNT: usize = 11;

/// Default reservoir size for multi-head ESN (scales with head count).
pub const ESN_MULTI_HEAD_RESERVOIR: usize = 500;

/// Default spectral radius for multi-head ESN.
pub const ESN_MULTI_HEAD_SPECTRAL_RADIUS: f32 = 0.95;

/// Default connectivity for multi-head ESN (sparser for larger reservoir).
pub const ESN_MULTI_HEAD_CONNECTIVITY: f32 = 0.05;

/// Configuration for Echo State Network
#[derive(Debug, Clone)]
pub struct ESNConfig {
    /// Number of input features
    pub input_size: usize,

    /// Number of reservoir neurons
    pub reservoir_size: usize,

    /// Number of output features (heads)
    pub output_size: usize,

    /// Target spectral radius (typically 0.9-0.99)
    pub spectral_radius: f32,

    /// Fraction of non-zero reservoir weights (0.0-1.0)
    pub connectivity: f32,

    /// Leak rate for temporal integration (0.0-1.0)
    pub leak_rate: f32,

    /// Ridge regression regularization parameter (> 0)
    pub regularization: f32,

    /// Random seed for reproducibility
    pub seed: u64,
}

impl ESNConfig {
    /// Create a multi-head ESN config (hotSpring 11-head standard).
    ///
    /// Produces an ESN with 11 output heads, a larger reservoir, and
    /// sparser connectivity tuned for multi-target prediction.
    pub fn multi_head(input_size: usize) -> Self {
        Self {
            input_size,
            reservoir_size: ESN_MULTI_HEAD_RESERVOIR,
            output_size: ESN_MULTI_HEAD_COUNT,
            spectral_radius: ESN_MULTI_HEAD_SPECTRAL_RADIUS,
            connectivity: ESN_MULTI_HEAD_CONNECTIVITY,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        }
    }
}

impl Default for ESNConfig {
    fn default() -> Self {
        Self {
            input_size: 1,
            reservoir_size: 100,
            output_size: 1,
            spectral_radius: 0.9,
            connectivity: 0.1,
            leak_rate: 0.3,
            regularization: 1e-6,
            seed: 42,
        }
    }
}

/// Validate ESN configuration parameters
pub fn validate_config(config: &ESNConfig) -> BarracudaResult<()> {
    let check = |cond: bool, msg: &str| -> BarracudaResult<()> {
        if cond {
            Ok(())
        } else {
            Err(BarracudaError::InvalidInput {
                message: msg.to_string(),
            })
        }
    };
    check(
        config.input_size > 0 && config.reservoir_size > 0 && config.output_size > 0,
        "All sizes must be greater than zero",
    )?;
    check(
        config.spectral_radius > 0.0 && config.spectral_radius <= 2.0,
        "Spectral radius must be in (0, 2]",
    )?;
    check(
        config.connectivity > 0.0 && config.connectivity <= 1.0,
        "Connectivity must be in (0, 1]",
    )?;
    check(
        config.leak_rate > 0.0 && config.leak_rate <= 1.0,
        "Leak rate must be in (0, 1]",
    )?;
    check(
        config.regularization > 0.0,
        "Regularization must be positive",
    )?;
    Ok(())
}

/// Check that a dimension matches expected size
pub fn expect_size(label: &str, expected: usize, actual: usize) -> BarracudaResult<()> {
    if actual == expected {
        return Ok(());
    }
    Err(BarracudaError::InvalidInput {
        message: format!("{label} size mismatch: expected {expected}, got {actual}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_config_accepts_valid_default() {
        let config = ESNConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn validate_config_rejects_zero_input_size() {
        let config = ESNConfig {
            input_size: 0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_zero_reservoir_size() {
        let config = ESNConfig {
            reservoir_size: 0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_zero_output_size() {
        let config = ESNConfig {
            output_size: 0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_spectral_radius_zero() {
        let config = ESNConfig {
            spectral_radius: 0.0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_spectral_radius_over_two() {
        let config = ESNConfig {
            spectral_radius: 2.1,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_connectivity_zero() {
        let config = ESNConfig {
            connectivity: 0.0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn validate_config_rejects_regularization_zero() {
        let config = ESNConfig {
            regularization: 0.0,
            ..Default::default()
        };
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn expect_size_ok_when_matching() {
        assert!(expect_size("test", 10, 10).is_ok());
    }

    #[test]
    fn expect_size_err_when_mismatch() {
        let err = expect_size("dim", 10, 5);
        assert!(err.is_err());
        let msg = format!("{:?}", err.unwrap_err());
        assert!(msg.contains("10") && msg.contains("5"));
    }

    #[test]
    fn multi_head_config_values() {
        let config = ESNConfig::multi_head(5);
        assert_eq!(config.input_size, 5);
        assert_eq!(config.reservoir_size, ESN_MULTI_HEAD_RESERVOIR);
        assert_eq!(config.output_size, ESN_MULTI_HEAD_COUNT);
        assert!(validate_config(&config).is_ok());
    }
}
