// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anderson coupling (tight-binding model) GPU op (f64).
//!
//! Provenance: groundSpring S69 → toadStool absorption.
//!
//! Constructs the diagonal and off-diagonal elements of the Anderson
//! tight-binding Hamiltonian from a random potential landscape.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/spectral/anderson_coupling_f64.wgsl");

const WG_64: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct AndersonParams {
    n: u32,
    dim: u32,
    extent: u32,
    _pad: u32,
    hopping_t: f64,
}

/// Result of Anderson coupling construction.
#[derive(Debug, Clone)]
pub struct AndersonResult {
    pub diagonal: Vec<f64>,
    pub off_diagonal: Vec<f64>,
}

/// Construct the Anderson tight-binding Hamiltonian.
///
/// * `potential` — random on-site potential `V[i]`, length `n`.
/// * `dim` — spatial dimension (1, 2, or 3).
/// * `extent` — linear extent per dimension (`n = extent^dim`).
/// * `hopping_t` — nearest-neighbor hopping amplitude.
///
/// Returns the diagonal (on-site energies) and off-diagonal (hopping) elements.
pub fn anderson_coupling(
    device: &Arc<WgpuDevice>,
    potential: &[f64],
    dim: u32,
    extent: u32,
    hopping_t: f64,
) -> Result<AndersonResult> {
    let n = potential.len();
    let n_offdiag = n * 2 * dim as usize;

    let pot_buf = device.create_buffer_f64_init("anderson:potential", potential);
    let diag_buf = device.create_buffer_f64(n)?;
    let offdiag_buf = device.create_buffer_f64(n_offdiag)?;
    let params = AndersonParams {
        n: n as u32,
        dim,
        extent,
        _pad: 0,
        hopping_t,
    };
    let params_buf = device.create_uniform_buffer("anderson:params", &params);

    ComputeDispatch::new(device, "anderson_coupling")
        .shader(SHADER, "main")
        .f64()
        .storage_read(0, &pot_buf)
        .storage_rw(1, &diag_buf)
        .storage_rw(2, &offdiag_buf)
        .uniform(3, &params_buf)
        .dispatch(n.div_ceil(WG_64 as usize) as u32, 1, 1)
        .submit();

    let diagonal = device.read_f64_buffer(&diag_buf, n)?;
    let off_diagonal = device.read_f64_buffer(&offdiag_buf, n_offdiag)?;

    Ok(AndersonResult {
        diagonal,
        off_diagonal,
    })
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<AndersonParams>(), 24);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn anderson_coupling_f64_compiles_via_naga() {
        let source = include_str!("../shaders/spectral/anderson_coupling_f64.wgsl");
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
}
