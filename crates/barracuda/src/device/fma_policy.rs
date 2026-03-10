// SPDX-License-Identifier: AGPL-3.0-only

//! FMA contraction policy for reproducibility-sensitive shaders.
//!
//! GPU compilers may fuse `a * b + c` into a single FMA instruction, which
//! changes the rounding behavior. For reproducibility-sensitive computations
//! (e.g., lattice QCD, gradient flow, BCS gap equations), FMA contraction
//! must be explicitly controlled.
//!
//! Cross-spring P1 (coralReef Iteration 30 ISSUE-011).

use serde::{Deserialize, Serialize};

/// FMA contraction policy for shader compilation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FmaPolicy {
    /// Allow the compiler to contract `a * b + c` → `fma(a, b, c)`.
    /// Best performance, may reduce reproducibility.
    Contract,
    /// Force splitting: `fma(a, b, c)` → `a * b + c` (separate multiply and add).
    /// Ensures bit-exact reproducibility across architectures.
    Separate,
    /// Use hardware default (compiler decides).
    #[default]
    Default,
}

impl FmaPolicy {
    /// Whether this policy requires explicit FMA splitting.
    #[must_use]
    pub fn requires_split(&self) -> bool {
        matches!(self, Self::Separate)
    }

    /// Whether this policy allows the compiler to contract freely.
    #[must_use]
    pub fn allows_contraction(&self) -> bool {
        matches!(self, Self::Contract | Self::Default)
    }
}

impl std::fmt::Display for FmaPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract => write!(f, "contract"),
            Self::Separate => write!(f, "separate"),
            Self::Default => write!(f, "default"),
        }
    }
}

/// Domains that require `FmaPolicy::Separate` for correctness.
///
/// These are physics domains where FMA contraction changes the rounding
/// pattern enough to cause measurable drift or fail validation.
#[must_use]
pub fn domain_requires_separate_fma(domain: &super::precision_tier::PhysicsDomain) -> bool {
    use super::precision_tier::PhysicsDomain;
    matches!(
        domain,
        PhysicsDomain::LatticeQcd | PhysicsDomain::GradientFlow | PhysicsDomain::NuclearEos
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fma_policy_display() {
        assert_eq!(FmaPolicy::Contract.to_string(), "contract");
        assert_eq!(FmaPolicy::Separate.to_string(), "separate");
        assert_eq!(FmaPolicy::Default.to_string(), "default");
    }

    #[test]
    fn test_fma_policy_default() {
        assert_eq!(FmaPolicy::default(), FmaPolicy::Default);
    }

    #[test]
    fn test_requires_split() {
        assert!(FmaPolicy::Separate.requires_split());
        assert!(!FmaPolicy::Contract.requires_split());
        assert!(!FmaPolicy::Default.requires_split());
    }

    #[test]
    fn test_domain_fma_requirements() {
        use super::super::precision_tier::PhysicsDomain;
        assert!(domain_requires_separate_fma(&PhysicsDomain::LatticeQcd));
        assert!(domain_requires_separate_fma(&PhysicsDomain::GradientFlow));
        assert!(!domain_requires_separate_fma(&PhysicsDomain::Statistics));
        assert!(!domain_requires_separate_fma(
            &PhysicsDomain::Bioinformatics
        ));
    }
}
