//! DF64 protein folding shaders for AlphaFold2-style structure prediction.
//!
//! These shaders use double-float (f32-pair) arithmetic for high precision on
//! FP32-only hardware. Compile via [`crate::device::WgpuDevice::compile_shader_df64`].

use crate::device::WgpuDevice;
use crate::error::Result;

// ── Shader sources ──

pub const WGSL_TORSION_ANGLES_DF64: &str =
    include_str!("../shaders/folding/torsion_angles_df64.wgsl");
pub const WGSL_DISTANCE_MATRIX_DF64: &str =
    include_str!("../shaders/folding/distance_matrix_df64.wgsl");
pub const WGSL_RMSD_DF64: &str = include_str!("../shaders/folding/rmsd_df64.wgsl");
pub const WGSL_CONTACT_MAP_DF64: &str = include_str!("../shaders/folding/contact_map_df64.wgsl");
pub const WGSL_LENNARD_JONES_DF64: &str =
    include_str!("../shaders/folding/lennard_jones_df64.wgsl");
pub const WGSL_COULOMB_ELECTROSTATIC_DF64: &str =
    include_str!("../shaders/folding/coulomb_electrostatic_df64.wgsl");
pub const WGSL_HYDROGEN_BOND_DF64: &str =
    include_str!("../shaders/folding/hydrogen_bond_df64.wgsl");
pub const WGSL_SOLVATION_ENERGY_DF64: &str =
    include_str!("../shaders/folding/solvation_energy_df64.wgsl");
pub const WGSL_GRADIENT_DESCENT_DF64: &str =
    include_str!("../shaders/folding/gradient_descent_df64.wgsl");
pub const WGSL_SIMULATED_ANNEALING_DF64: &str =
    include_str!("../shaders/folding/simulated_annealing_df64.wgsl");
pub const WGSL_BACKBONE_RESTRAINTS_DF64: &str =
    include_str!("../shaders/folding/backbone_restraints_df64.wgsl");
pub const WGSL_SIDE_CHAIN_PACKING_DF64: &str =
    include_str!("../shaders/folding/side_chain_packing_df64.wgsl");
pub const WGSL_MSA_ATTENTION_DF64: &str =
    include_str!("../shaders/folding/msa_attention_df64.wgsl");
pub const WGSL_PAIR_REPRESENTATION_DF64: &str =
    include_str!("../shaders/folding/pair_representation_df64.wgsl");
pub const WGSL_STRUCTURE_MODULE_DF64: &str =
    include_str!("../shaders/folding/structure_module_df64.wgsl");

/// Protein folding DF64 shader operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FoldingOp {
    /// Phi/psi/omega backbone torsion angles from Cα coordinates.
    TorsionAngles,
    /// Pairwise Cα distance matrix.
    DistanceMatrix,
    /// Root Mean Square Deviation (squared deviations for reduction).
    Rmsd,
    /// Binary contact map from distance matrix with threshold.
    ContactMap,
    /// 12-6 Lennard-Jones potential for van der Waals.
    LennardJones,
    /// Coulomb electrostatic energy with dielectric.
    CoulombElectrostatic,
    /// Hydrogen bond energy (donor-acceptor geometry).
    HydrogenBond,
    /// Implicit solvation (GBSA-style, per-atom).
    SolvationEnergy,
    /// Steepest descent energy minimization step.
    GradientDescent,
    /// Simulated annealing Metropolis acceptance.
    SimulatedAnnealing,
    /// Harmonic restraints on backbone atoms.
    BackboneRestraints,
    /// Rotamer scoring for side chain placement.
    SideChainPacking,
    /// MSA row/column attention scores.
    MsaAttention,
    /// Pair features update (outer product mean).
    PairRepresentation,
    /// IPA (Invariant Point Attention) frame update.
    StructureModule,
}

impl FoldingOp {
    /// WGSL source for this operation.
    pub fn source(&self) -> &'static str {
        match self {
            FoldingOp::TorsionAngles => WGSL_TORSION_ANGLES_DF64,
            FoldingOp::DistanceMatrix => WGSL_DISTANCE_MATRIX_DF64,
            FoldingOp::Rmsd => WGSL_RMSD_DF64,
            FoldingOp::ContactMap => WGSL_CONTACT_MAP_DF64,
            FoldingOp::LennardJones => WGSL_LENNARD_JONES_DF64,
            FoldingOp::CoulombElectrostatic => WGSL_COULOMB_ELECTROSTATIC_DF64,
            FoldingOp::HydrogenBond => WGSL_HYDROGEN_BOND_DF64,
            FoldingOp::SolvationEnergy => WGSL_SOLVATION_ENERGY_DF64,
            FoldingOp::GradientDescent => WGSL_GRADIENT_DESCENT_DF64,
            FoldingOp::SimulatedAnnealing => WGSL_SIMULATED_ANNEALING_DF64,
            FoldingOp::BackboneRestraints => WGSL_BACKBONE_RESTRAINTS_DF64,
            FoldingOp::SideChainPacking => WGSL_SIDE_CHAIN_PACKING_DF64,
            FoldingOp::MsaAttention => WGSL_MSA_ATTENTION_DF64,
            FoldingOp::PairRepresentation => WGSL_PAIR_REPRESENTATION_DF64,
            FoldingOp::StructureModule => WGSL_STRUCTURE_MODULE_DF64,
        }
    }

    /// Shader label for debugging.
    pub fn label(&self) -> &'static str {
        match self {
            FoldingOp::TorsionAngles => "torsion_angles_df64",
            FoldingOp::DistanceMatrix => "distance_matrix_df64",
            FoldingOp::Rmsd => "rmsd_df64",
            FoldingOp::ContactMap => "contact_map_df64",
            FoldingOp::LennardJones => "lennard_jones_df64",
            FoldingOp::CoulombElectrostatic => "coulomb_electrostatic_df64",
            FoldingOp::HydrogenBond => "hydrogen_bond_df64",
            FoldingOp::SolvationEnergy => "solvation_energy_df64",
            FoldingOp::GradientDescent => "gradient_descent_df64",
            FoldingOp::SimulatedAnnealing => "simulated_annealing_df64",
            FoldingOp::BackboneRestraints => "backbone_restraints_df64",
            FoldingOp::SideChainPacking => "side_chain_packing_df64",
            FoldingOp::MsaAttention => "msa_attention_df64",
            FoldingOp::PairRepresentation => "pair_representation_df64",
            FoldingOp::StructureModule => "structure_module_df64",
        }
    }
}

/// Compile a folding DF64 shader for the given device and operation.
pub fn compile_folding_shader(device: &WgpuDevice, op: FoldingOp) -> Result<wgpu::ShaderModule> {
    let source = op.source();
    let label = op.label();
    Ok(device.compile_shader_df64(source, Some(label)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_sync;

    #[test]
    fn test_folding_shaders_compile() {
        let device = get_test_device_sync();
        let ops = [
            FoldingOp::TorsionAngles,
            FoldingOp::DistanceMatrix,
            FoldingOp::Rmsd,
            FoldingOp::ContactMap,
            FoldingOp::LennardJones,
            FoldingOp::CoulombElectrostatic,
            FoldingOp::HydrogenBond,
            FoldingOp::SolvationEnergy,
            FoldingOp::GradientDescent,
            FoldingOp::SimulatedAnnealing,
            FoldingOp::BackboneRestraints,
            FoldingOp::SideChainPacking,
            FoldingOp::MsaAttention,
            FoldingOp::PairRepresentation,
            FoldingOp::StructureModule,
        ];
        for op in ops {
            let result = compile_folding_shader(&device, op);
            assert!(result.is_ok(), "{} shader failed to compile", op.label());
        }
    }
}
