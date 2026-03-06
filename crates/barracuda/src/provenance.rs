// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-Spring Provenance Tags (L-005)
//!
//! Metadata for absorbed code by capability domain (`lattice_qcd`, `marine_bio`,
//! `ml_inference`, etc.). Enables traceability of feature origins across the
//! ecoPrimals ecosystem without hardcoding primal identities.

/// Provenance metadata for absorbed items.
///
/// Origin names are capability-based identifiers (e.g. "`lattice_qcd`",
/// "`marine_bio`", "`ml_inference`") — not primal names. barraCuda describes
/// capabilities and discovers peers at runtime via capability matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceTag {
    /// Capability domain (e.g. "`lattice_qcd`", "`marine_bio`", "`ml_inference`")
    pub origin: &'static str,
    /// Absorbed session or handoff identifier
    pub absorbed_session: &'static str,
    /// Human-readable description
    pub description: &'static str,
}

impl ProvenanceTag {
    /// Validates that all fields are structurally correct.
    ///
    /// Checks non-empty strings and ASCII-safe origin names.
    /// Does NOT check against a hardcoded allowlist — any valid capability identifier is accepted.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.origin.is_empty()
            && !self.absorbed_session.is_empty()
            && !self.description.is_empty()
            && self
                .origin
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    }
}

// ─── lattice_qcd / physics_validation provenance ─────────────────────────────

/// CG (Conjugate Gradient) shaders for lattice QCD.
pub const PROV_CG_SHADERS: ProvenanceTag = ProvenanceTag {
    origin: "lattice_qcd",
    absorbed_session: "lattice/cg v0.6.1",
    description: "Conjugate gradient solver for lattice QCD",
};

/// Lattice QCD SU3 and Dirac operations.
pub const PROV_LATTICE_QCD: ProvenanceTag = ProvenanceTag {
    origin: "lattice_qcd",
    absorbed_session: "lattice v0.6.4",
    description: "Lattice QCD gauge field theory operations",
};

/// Spectral theory: Lanczos, Anderson localization, Hofstadter.
pub const PROV_SPECTRAL: ProvenanceTag = ProvenanceTag {
    origin: "physics_validation",
    absorbed_session: "spectral v0.6.0",
    description: "Spectral theory and eigenvalue methods",
};

/// Hermite and Laguerre polynomials for nuclear physics.
pub const PROV_HERMITE_LAGUERRE: ProvenanceTag = ProvenanceTag {
    origin: "physics_validation",
    absorbed_session: "special v0.6",
    description: "Hermite/Laguerre polynomials for nuclear physics",
};

// ─── marine_bio / diversity_stats provenance ─────────────────────────────────

/// ESN NPU backend and reservoir design.
pub const PROV_ESN_NPU: ProvenanceTag = ProvenanceTag {
    origin: "ml_inference",
    absorbed_session: "esn_v2 npu",
    description: "Echo State Network NPU backend",
};

/// Generic ODE solver replacing domain-specific shaders.
pub const PROV_ODE_GENERIC: ProvenanceTag = ProvenanceTag {
    origin: "marine_bio",
    absorbed_session: "numerical/ode_generic",
    description: "Generic ODE solver (phage_defense, cooperation, etc.)",
};

/// Bio/life-science GPU primitives.
pub const PROV_BIO_PRIMITIVES: ProvenanceTag = ProvenanceTag {
    origin: "marine_bio",
    absorbed_session: "bio handoff v4",
    description: "Life-science GPU primitives (Felsenstein, Smith-Waterman, etc.)",
};

/// Bray-Curtis distance for metagenomics.
pub const PROV_BRAY_CURTIS: ProvenanceTag = ProvenanceTag {
    origin: "diversity_stats",
    absorbed_session: "bray_curtis_f64",
    description: "Bray-Curtis distance for diversity metrics",
};

// ─── ml_inference / neural_dispatch provenance ──────────────────────────────

/// Tensor session batching (S-01, S-11 handoffs).
pub const PROV_TENSOR_SESSION: ProvenanceTag = ProvenanceTag {
    origin: "neural_dispatch",
    absorbed_session: "S-01, S-11",
    description: "TensorSession operation batching",
};

/// Swarm NN forward and scores (Paper 015).
pub const PROV_SWARM_NN: ProvenanceTag = ProvenanceTag {
    origin: "ml_inference",
    absorbed_session: "metalForge swarm",
    description: "Swarm neural network forward and scores",
};

/// Batch fitness evaluation.
pub const PROV_BATCH_FITNESS: ProvenanceTag = ProvenanceTag {
    origin: "ml_inference",
    absorbed_session: "metalForge batch_fitness",
    description: "Batched fitness evaluation for evolutionary algorithms",
};

/// RK45 adaptive ODE for regulatory networks.
pub const PROV_RK45_ADAPTIVE: ProvenanceTag = ProvenanceTag {
    origin: "neural_dispatch",
    absorbed_session: "metalForge rk45",
    description: "Adaptive Dormand-Prince RK45 for regulatory networks",
};

/// All provenance tags for enumeration.
pub const ALL_TAGS: &[ProvenanceTag] = &[
    PROV_CG_SHADERS,
    PROV_LATTICE_QCD,
    PROV_SPECTRAL,
    PROV_HERMITE_LAGUERRE,
    PROV_ESN_NPU,
    PROV_ODE_GENERIC,
    PROV_BIO_PRIMITIVES,
    PROV_BRAY_CURTIS,
    PROV_TENSOR_SESSION,
    PROV_SWARM_NN,
    PROV_BATCH_FITNESS,
    PROV_RK45_ADAPTIVE,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tags_are_structurally_valid() {
        for tag in ALL_TAGS {
            assert!(
                tag.is_valid(),
                "structurally invalid provenance tag: {tag:?}"
            );
        }
    }

    #[test]
    fn tag_count() {
        assert_eq!(ALL_TAGS.len(), 12);
    }
}
