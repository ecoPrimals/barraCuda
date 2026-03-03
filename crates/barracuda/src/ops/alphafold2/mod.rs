// SPDX-License-Identifier: AGPL-3.0-only

//! AlphaFold2 Evoformer GPU primitives (f64).
//!
//! Provenance: neuralSpring S69 → toadStool absorption.
//!
//! | Function | Shader | Primitive |
//! |----------|--------|-----------|
//! | `triangle_mul_outgoing` | `triangle_mul_outgoing_f64.wgsl` | Outgoing triangle multiplication |
//! | `triangle_mul_incoming` | `triangle_mul_incoming_f64.wgsl` | Incoming triangle multiplication |
//! | `msa_row_attention`     | `msa_row_attention_scores_f64.wgsl` | MSA row-wise attention scores |
//! | `msa_col_attention`     | `msa_col_attention_scores_f64.wgsl` | MSA column-wise attention scores |
//! | `ipa_scores`            | `ipa_scores_f64.wgsl`            | Invariant Point Attention scores |
//! | `triangle_attention`    | `triangle_attention_f64.wgsl`    | Triangle self-attention |
//! | `backbone_update`       | `backbone_update_f64.wgsl`       | SE(3) backbone frame update |
//! | `torsion_angles`        | `torsion_angles_f64.wgsl`        | Dihedral angle extraction |
//! | `outer_product_mean`    | `outer_product_mean_f64.wgsl`   | Pair representation OPM update |
//! | `pair_transition`       | `pair_transition_f64.wgsl`      | Pair representation 2-layer MLP |
//! | `template_embedding`    | `template_embedding_f64.wgsl`   | Template stack averaging |
//! | `recycling_update`     | `recycling_update_f64.wgsl`     | Recycling iteration with layer norm |
//! | `fape_loss`            | `fape_loss_f64.wgsl`            | Frame Aligned Point Error loss |
//! | `plddt`                 | `plddt_f64.wgsl`                | Predicted LDDT confidence |
//! | `structure_violation`   | `structure_violation_f64.wgsl`  | Steric clash + bond violations |
//! | `confidence`            | `confidence_f64.wgsl`           | Per-residue confidence from logits |
//! | `ensemble_average`      | `ensemble_average_f64.wgsl`     | Ensemble position averaging |

pub(crate) const WG_64: u32 = 64;

pub mod attention;
pub mod embedding;
pub mod structure;
pub mod triangle;

// Re-export all public functions for unchanged import path
pub use attention::{ipa_scores, msa_col_attention, msa_row_attention, triangle_attention};
pub use embedding::{
    ensemble_average, outer_product_mean, pair_transition, recycling_update, template_embedding,
};
pub use structure::{
    backbone_update, confidence, fape_loss, plddt, structure_violation, torsion_angles,
};
pub use triangle::{triangle_mul_incoming, triangle_mul_outgoing};
