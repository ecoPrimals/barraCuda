// SPDX-License-Identifier: AGPL-3.0-only
//! Parameter structs for the 5 biological ODE systems.
//!
//! Each struct carries the kinetic constants for one model and provides
//! `to_flat()` / `from_flat()` for GPU buffer packing. The flat layout
//! matches the `params` array indices in the WGSL derivative functions.

mod bistable;
mod capacitor;
mod cooperation;
mod multi_signal;
mod phage_defense;
mod qs_biofilm;

pub use bistable::{BISTABLE_N_PARAMS, BISTABLE_N_VARS, BistableParams};
pub use capacitor::{CAPACITOR_N_PARAMS, CAPACITOR_N_VARS, CapacitorParams};
pub use cooperation::{COOPERATION_N_PARAMS, COOPERATION_N_VARS, CooperationParams};
pub use multi_signal::{MULTI_SIGNAL_N_PARAMS, MULTI_SIGNAL_N_VARS, MultiSignalParams};
pub use phage_defense::{PHAGE_DEFENSE_N_PARAMS, PHAGE_DEFENSE_N_VARS, PhageDefenseParams};
pub use qs_biofilm::QsBiofilmParams;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qs_biofilm_roundtrip() {
        let params = QsBiofilmParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), super::qs_biofilm::QS_BIOFILM_N_PARAMS);
        let restored = QsBiofilmParams::from_flat(&flat);
        assert!((restored.mu_max - params.mu_max).abs() < f64::EPSILON);
        assert!((restored.d_bio - params.d_bio).abs() < f64::EPSILON);
    }

    #[test]
    fn qs_biofilm_default_positive() {
        let params = QsBiofilmParams::default();
        let flat = params.to_flat();
        for (i, &v) in flat.iter().enumerate() {
            assert!(v > 0.0, "param[{i}] must be positive, got {v}");
        }
    }

    #[test]
    fn bistable_roundtrip() {
        let params = BistableParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), BISTABLE_N_PARAMS);
        let restored = BistableParams::from_flat(&flat);
        assert!((restored.base.mu_max - params.base.mu_max).abs() < f64::EPSILON);
        assert!((restored.alpha_fb - params.alpha_fb).abs() < f64::EPSILON);
    }

    #[test]
    fn capacitor_roundtrip() {
        let params = CapacitorParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), CAPACITOR_N_PARAMS);
        let restored = CapacitorParams::from_flat(&flat);
        assert!((restored.mu_max - params.mu_max).abs() < f64::EPSILON);
        assert!((restored.d_mot - params.d_mot).abs() < f64::EPSILON);
    }

    #[test]
    fn cooperation_roundtrip() {
        let params = CooperationParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), COOPERATION_N_PARAMS);
        let restored = CooperationParams::from_flat(&flat);
        assert!((restored.mu_coop - params.mu_coop).abs() < f64::EPSILON);
        assert!((restored.d_bio - params.d_bio).abs() < f64::EPSILON);
    }

    #[test]
    fn multi_signal_roundtrip() {
        let params = MultiSignalParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), MULTI_SIGNAL_N_PARAMS);
        let restored = MultiSignalParams::from_flat(&flat);
        assert!((restored.mu_max - params.mu_max).abs() < f64::EPSILON);
        assert!((restored.d_bio - params.d_bio).abs() < f64::EPSILON);
    }

    #[test]
    fn phage_defense_roundtrip() {
        let params = PhageDefenseParams::default();
        let flat = params.to_flat();
        assert_eq!(flat.len(), PHAGE_DEFENSE_N_PARAMS);
        let restored = PhageDefenseParams::from_flat(&flat);
        assert!((restored.mu_max - params.mu_max).abs() < f64::EPSILON);
        assert!((restored.burst_size - params.burst_size).abs() < f64::EPSILON);
    }
}
