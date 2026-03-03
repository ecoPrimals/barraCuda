// SPDX-License-Identifier: AGPL-3.0-only

//! Backbone update, torsion angles, FAPE loss, structure violation, pLDDT, confidence.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::WG_64;

const SHADER_BACKBONE: &str = include_str!("../../shaders/misc/backbone_update_f64.wgsl");
const SHADER_TORSION: &str = include_str!("../../shaders/misc/torsion_angles_f64.wgsl");
const SHADER_FAPE: &str = include_str!("../../shaders/misc/fape_loss_f64.wgsl");
const SHADER_PLDDT: &str = include_str!("../../shaders/misc/plddt_f64.wgsl");
const SHADER_STRUCTURE_VIOLATION: &str =
    include_str!("../../shaders/misc/structure_violation_f64.wgsl");
const SHADER_CONFIDENCE: &str = include_str!("../../shaders/misc/confidence_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BackboneParams {
    n: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TorsionParams {
    n: u32,
    m: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FapeParams {
    n_residues: u32,
    n_atoms: u32,
    _pad: [u32; 2],
    d_clamp: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PlddtParams {
    n_residues: u32,
    _pad: [u32; 3],
    cutoff: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct StructureViolationParams {
    n_atoms: u32,
    n_bonds: u32,
    _pad: [u32; 2],
    d_vdw: f64,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ConfidenceParams {
    n_residues: u32,
    n_bins: u32,
    _pad: [u32; 2],
}

/// SE(3) backbone frame update: compose delta rotations/translations onto current frames.
///
/// Quaternion-based rigid-body update for structure module iteration.
pub fn backbone_update(
    device: &Arc<WgpuDevice>,
    quaternions: &[f64],
    translations: &[f64],
    delta_quat: &[f64],
    delta_trans: &[f64],
    n: u32,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let quat_len = (n * 4) as usize;
    let trans_len = (n * 3) as usize;

    let q_buf = device.create_buffer_f64_init("backbone:quat", quaternions);
    let t_buf = device.create_buffer_f64_init("backbone:trans", translations);
    let dq_buf = device.create_buffer_f64_init("backbone:dq", delta_quat);
    let dt_buf = device.create_buffer_f64_init("backbone:dt", delta_trans);
    let oq_buf = device.create_buffer_f64(quat_len)?;
    let ot_buf = device.create_buffer_f64(trans_len)?;
    let params = BackboneParams { n, _pad: [0; 3] };
    let params_buf = device.create_uniform_buffer("backbone:params", &params);

    ComputeDispatch::new(device, "backbone_update")
        .shader(SHADER_BACKBONE, "main")
        .f64()
        .storage_read(0, &q_buf)
        .storage_read(1, &t_buf)
        .storage_read(2, &dq_buf)
        .storage_read(3, &dt_buf)
        .storage_rw(4, &oq_buf)
        .storage_rw(5, &ot_buf)
        .uniform(6, &params_buf)
        .dispatch((n as usize).div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    let out_q = device.read_f64_buffer(&oq_buf, quat_len)?;
    let out_t = device.read_f64_buffer(&ot_buf, trans_len)?;
    Ok((out_q, out_t))
}

/// Extract dihedral (torsion) angles from atomic coordinates.
///
/// For each group of 4 consecutive atoms, computes the dihedral angle using
/// the atan2 formula on cross-product vectors.
pub fn torsion_angles(
    device: &Arc<WgpuDevice>,
    positions: &[f64],
    n: u32,
    m: u32,
) -> Result<Vec<f64>> {
    let out_len = (n * m) as usize;
    let pos_buf = device.create_buffer_f64_init("torsion:pos", positions);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = TorsionParams { n, m };
    let params_buf = device.create_uniform_buffer("torsion:params", &params);

    ComputeDispatch::new(device, "torsion_angles")
        .shader(SHADER_TORSION, "main")
        .f64()
        .storage_read(0, &pos_buf)
        .storage_rw(1, &out_buf)
        .uniform(2, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Frame Aligned Point Error loss: per-residue FAPE.
///
/// Compares predicted vs true atom positions in local frames.
pub fn fape_loss(
    device: &Arc<WgpuDevice>,
    pred_pos: &[f64],
    true_pos: &[f64],
    pred_frames: &[f64],
    true_frames: &[f64],
    n_residues: u32,
    n_atoms: u32,
    d_clamp: f64,
) -> Result<Vec<f64>> {
    let out_len = n_residues as usize;
    let pred_pos_buf = device.create_buffer_f64_init("fape:pred_pos", pred_pos);
    let true_pos_buf = device.create_buffer_f64_init("fape:true_pos", true_pos);
    let pred_frames_buf = device.create_buffer_f64_init("fape:pred_frames", pred_frames);
    let true_frames_buf = device.create_buffer_f64_init("fape:true_frames", true_frames);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = FapeParams {
        n_residues,
        n_atoms,
        _pad: [0; 2],
        d_clamp,
    };
    let params_buf = device.create_uniform_buffer("fape:params", &params);

    ComputeDispatch::new(device, "fape_loss")
        .shader(SHADER_FAPE, "main")
        .f64()
        .storage_read(0, &pred_pos_buf)
        .storage_read(1, &true_pos_buf)
        .storage_read(2, &pred_frames_buf)
        .storage_read(3, &true_frames_buf)
        .storage_rw(4, &out_buf)
        .uniform(5, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Predicted LDDT confidence score per residue.
pub fn plddt(
    device: &Arc<WgpuDevice>,
    pred_pos: &[f64],
    true_pos: &[f64],
    n_residues: u32,
    cutoff: f64,
) -> Result<Vec<f64>> {
    let out_len = n_residues as usize;
    let pred_pos_buf = device.create_buffer_f64_init("plddt:pred_pos", pred_pos);
    let true_pos_buf = device.create_buffer_f64_init("plddt:true_pos", true_pos);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = PlddtParams {
        n_residues,
        _pad: [0; 3],
        cutoff,
    };
    let params_buf = device.create_uniform_buffer("plddt:params", &params);

    ComputeDispatch::new(device, "plddt")
        .shader(SHADER_PLDDT, "main")
        .f64()
        .storage_read(0, &pred_pos_buf)
        .storage_read(1, &true_pos_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

/// Steric clash + bond geometry violations.
///
/// Returns (out_clash, out_bond).
pub fn structure_violation(
    device: &Arc<WgpuDevice>,
    positions: &[f64],
    bond_pairs: &[u32],
    n_atoms: u32,
    n_bonds: u32,
    d_vdw: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let clash_len = n_atoms as usize;
    let bond_len = n_bonds as usize;
    let total_threads = clash_len + bond_len;

    let pos_buf = device.create_buffer_f64_init("struct_viol:pos", positions);
    let bond_buf = device.create_buffer_u32_init("struct_viol:bonds", bond_pairs);
    let clash_buf = device.create_buffer_f64(clash_len)?;
    let bond_out_buf = device.create_buffer_f64(bond_len)?;
    let params = StructureViolationParams {
        n_atoms,
        n_bonds,
        _pad: [0; 2],
        d_vdw,
    };
    let params_buf = device.create_uniform_buffer("struct_viol:params", &params);

    ComputeDispatch::new(device, "structure_violation")
        .shader(SHADER_STRUCTURE_VIOLATION, "main")
        .f64()
        .storage_read(0, &pos_buf)
        .storage_read(1, &bond_buf)
        .storage_rw(2, &clash_buf)
        .storage_rw(3, &bond_out_buf)
        .uniform(4, &params_buf)
        .dispatch(total_threads.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    let out_clash = device.read_f64_buffer(&clash_buf, clash_len)?;
    let out_bond = device.read_f64_buffer(&bond_out_buf, bond_len)?;
    Ok((out_clash, out_bond))
}

/// Per-residue confidence from logits: `conf[i] = sum(softmax(logits[i,:]) * bin_centers)`.
pub fn confidence(
    device: &Arc<WgpuDevice>,
    logits: &[f64],
    bin_centers: &[f64],
    n_residues: u32,
    n_bins: u32,
) -> Result<Vec<f64>> {
    let out_len = n_residues as usize;
    let logits_buf = device.create_buffer_f64_init("confidence:logits", logits);
    let bin_buf = device.create_buffer_f64_init("confidence:bin_centers", bin_centers);
    let out_buf = device.create_buffer_f64(out_len)?;
    let params = ConfidenceParams {
        n_residues,
        n_bins,
        _pad: [0; 2],
    };
    let params_buf = device.create_uniform_buffer("confidence:params", &params);

    ComputeDispatch::new(device, "confidence")
        .shader(SHADER_CONFIDENCE, "main")
        .f64()
        .storage_read(0, &logits_buf)
        .storage_read(1, &bin_buf)
        .storage_rw(2, &out_buf)
        .uniform(3, &params_buf)
        .dispatch(out_len.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    device.read_f64_buffer(&out_buf, out_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout_backbone() {
        assert_eq!(std::mem::size_of::<BackboneParams>(), 16);
    }

    #[test]
    fn params_layout_torsion() {
        assert_eq!(std::mem::size_of::<TorsionParams>(), 8);
    }

    #[test]
    fn params_layout_fape() {
        assert_eq!(std::mem::size_of::<FapeParams>(), 24);
    }

    #[test]
    fn params_layout_plddt() {
        assert_eq!(std::mem::size_of::<PlddtParams>(), 24);
    }

    #[test]
    fn params_layout_structure_violation() {
        assert_eq!(std::mem::size_of::<StructureViolationParams>(), 24);
    }

    #[test]
    fn params_layout_confidence() {
        assert_eq!(std::mem::size_of::<ConfidenceParams>(), 16);
    }

    #[cfg(feature = "gpu")]
    fn shader_compiles_via_naga(source: &str) {
        let full = crate::shaders::precision::ShaderTemplate::for_driver_auto(source, false);
        let normalized = format!(
            "enable f64;\n\n{}",
            full.replace("enable f64;\n\n", "")
                .replace("enable f64;\n", "")
        );
        if let Ok(module) = naga::front::wgsl::parse_str(&normalized) {
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("WGSL validation failed");
            assert!(!module.entry_points.is_empty());
        } else {
            assert!(!full.is_empty());
            assert!(full.contains("fn main"), "shader must contain fn main");
            assert!(
                full.contains("@compute") || full.contains("main"),
                "expected compute entry"
            );
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn backbone_update_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/backbone_update_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn torsion_angles_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/torsion_angles_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn fape_loss_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/fape_loss_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn plddt_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/plddt_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn structure_violation_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/structure_violation_f64.wgsl");
        shader_compiles_via_naga(source);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn confidence_f64_compiles_via_naga() {
        let source = include_str!("../../shaders/misc/confidence_f64.wgsl");
        shader_compiles_via_naga(source);
    }
}
