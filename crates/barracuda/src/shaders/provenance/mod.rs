// SPDX-License-Identifier: AGPL-3.0-only
//! Shader provenance tracking — cross-spring evolution registry.
//!
//! Tracks which spring domain each WGSL shader originated from, which
//! springs currently consume it, and how patterns flow across domains.
//! This is the programmatic equivalent of the provenance comments in shader
//! headers, enabling runtime introspection and evolution auditing.
//!
//! # Cross-Spring Evolution (Write → Absorb → Lean)
//!
//! Shaders flow between springs following the ecosystem pattern:
//! - **Write**: a spring creates a domain-specific shader
//! - **Absorb**: barraCuda generalises it as a reusable primitive
//! - **Lean**: other springs consume the upstream version
//!
//! # Module structure
//!
//! - [`types`] — domain enums and provenance record structs
//! - [`registry`] — the canonical shader registry, evolution timeline, and query functions
//! - [`report`] — markdown report generation

pub mod registry;
pub mod report;
pub mod types;

pub use registry::{
    EVOLUTION_TIMELINE, REGISTRY, cross_spring_matrix, cross_spring_shaders, shaders_consumed_by,
    shaders_from,
};
pub use report::evolution_report;
pub use types::{EvolutionEvent, ShaderCategory, ShaderRecord, SpringDomain};
