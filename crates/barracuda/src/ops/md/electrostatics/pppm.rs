// SPDX-License-Identifier: AGPL-3.0-or-later
//! PPPM Electrostatics Orchestration
//!
//! Combines all PPPM components into a complete long-range electrostatics solver.
//!
//! # Algorithm
//!
//! ```text
//! 1. Spread charges to mesh (CPU, B-spline interpolation)
//! 2. Forward FFT: ρ(r) → ρ̃(k)
//! 3. Apply Green's function: φ̃(k) = G(k) × ρ̃(k)
//! 4. Backward FFT: φ̃(k) → φ(r)
//! 5. Interpolate forces: F = -q∇φ (CPU, B-spline gradient)
//! 6. Add short-range: F += F_short (CPU, erfc-damped)
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use barracuda::ops::md::electrostatics::{Pppm, PppmParams, PppmAccuracy};
//!
//! // Create PPPM solver
//! let params = PppmParams::auto(n_particles, box_side, PppmAccuracy::Medium);
//! let pppm = Pppm::new(params);
//!
//! // Compute forces and energy
//! let (forces, energy) = pppm.compute(&positions, &charges)?;
//! ```

#[cfg(test)]
use std::f64::consts::PI;
use std::sync::Arc;

use crate::device::WgpuDevice;
use crate::ops::fft::Fft3DF64;

use super::{
    compute_short_range, dipole_correction, interpolate_forces, self_energy_correction,
    spread_charges_with_coeffs, ChargeMesh, GreensFunction, PotentialMesh, PppmParams,
};

/// PPPM Electrostatics Solver
///
/// Provides efficient O(N log N) electrostatics for periodic systems.
/// Always uses GPU FFT for k-space solves.
#[derive(Clone)]
pub struct Pppm {
    /// PPPM configuration parameters
    params: PppmParams,

    /// Precomputed Green's function G(k)
    greens: GreensFunction,

    /// GPU device for FFT execution
    device: Arc<WgpuDevice>,
}

impl Pppm {
    /// Create a new PPPM solver with given parameters and GPU device
    pub fn new(device: Arc<WgpuDevice>, params: PppmParams) -> Self {
        let greens = GreensFunction::new(&params);
        Self {
            params,
            greens,
            device,
        }
    }

    /// Compute electrostatic forces and total energy
    ///
    /// # Arguments
    /// * `positions` - Particle positions [x, y, z]
    /// * `charges` - Particle charges
    ///
    /// # Returns
    /// (forces, total_energy) where forces[i] = [fx, fy, fz]
    pub fn compute(
        &self,
        positions: &[[f64; 3]],
        charges: &[f64],
    ) -> Result<(Vec<[f64; 3]>, f64), PppmError> {
        if positions.len() != charges.len() {
            return Err(PppmError::SizeMismatch {
                positions: positions.len(),
                charges: charges.len(),
            });
        }

        let n = positions.len();
        if n == 0 {
            return Ok((vec![], 0.0));
        }

        // 1. Spread charges to mesh (returns coefficients for force interpolation)
        let (charge_mesh, coeffs) = spread_charges_with_coeffs(positions, charges, &self.params);

        // 2. Forward FFT: ρ(r) → ρ̃(k)
        let rho_k = self.forward_fft(&charge_mesh)?;

        // 3. Apply Green's function: φ̃(k) = G(k) × ρ̃(k)
        let phi_k = self.greens.apply(&rho_k);

        // 4. Compute k-space energy
        let volume = self.params.box_dims[0] * self.params.box_dims[1] * self.params.box_dims[2];
        let e_kspace = self.greens.kspace_energy(&rho_k, volume);

        // 5. Backward FFT: φ̃(k) → φ(r)
        let potential_mesh = self.backward_fft(&phi_k)?;

        // 6. Interpolate k-space forces from mesh
        let kspace_forces = interpolate_forces(&potential_mesh, charges, &coeffs, &self.params);

        // 7. Compute short-range forces and energy
        let (short_forces, e_short) = compute_short_range(positions, charges, &self.params);

        // 8. Compute corrections
        let e_self =
            self_energy_correction(charges, self.params.alpha, self.params.coulomb_constant);
        let e_dipole = dipole_correction(
            positions,
            charges,
            self.params.box_dims,
            self.params.coulomb_constant,
        );

        // 9. Combine forces
        let mut forces = vec![[0.0, 0.0, 0.0]; n];
        for i in 0..n {
            forces[i][0] = kspace_forces[i][0] + short_forces[i][0];
            forces[i][1] = kspace_forces[i][1] + short_forces[i][1];
            forces[i][2] = kspace_forces[i][2] + short_forces[i][2];
        }

        // 10. Total energy
        let total_energy = e_kspace + e_short + e_self + e_dipole;

        Ok((forces, total_energy))
    }

    /// Compute forces only (slightly faster if energy not needed)
    pub fn compute_forces(
        &self,
        positions: &[[f64; 3]],
        charges: &[f64],
    ) -> Result<Vec<[f64; 3]>, PppmError> {
        let (forces, _) = self.compute(positions, charges)?;
        Ok(forces)
    }

    /// Get the PPPM parameters
    pub fn params(&self) -> &PppmParams {
        &self.params
    }

    /// Forward FFT: convert charge mesh to k-space (GPU)
    async fn forward_fft_async(&self, mesh: &ChargeMesh) -> Result<Vec<f64>, PppmError> {
        let [kx, ky, kz] = self.params.mesh_dims;

        if !kx.is_power_of_two() || !ky.is_power_of_two() || !kz.is_power_of_two() {
            return Err(PppmError::FftError(format!(
                "Mesh dimensions must be powers of 2 for GPU FFT, got ({kx}, {ky}, {kz})"
            )));
        }

        let size = kx * ky * kz;

        let mut complex = vec![0.0f64; size * 2];
        for i in 0..size {
            complex[i * 2] = mesh.values[i];
        }

        let fft = Fft3DF64::new(self.device.clone(), kx, ky, kz)
            .map_err(|e| PppmError::FftError(e.to_string()))?;

        fft.forward(&complex)
            .await
            .map_err(|e| PppmError::FftError(e.to_string()))
    }

    fn forward_fft(&self, mesh: &ChargeMesh) -> Result<Vec<f64>, PppmError> {
        crate::device::test_pool::tokio_block_on(self.forward_fft_async(mesh))
    }

    async fn backward_fft_async(&self, phi_k: &[f64]) -> Result<PotentialMesh, PppmError> {
        let [kx, ky, kz] = self.params.mesh_dims;

        if !kx.is_power_of_two() || !ky.is_power_of_two() || !kz.is_power_of_two() {
            return Err(PppmError::FftError(format!(
                "Mesh dimensions must be powers of 2 for GPU FFT, got ({kx}, {ky}, {kz})"
            )));
        }

        let size = kx * ky * kz;

        let fft = Fft3DF64::new(self.device.clone(), kx, ky, kz)
            .map_err(|e| PppmError::FftError(e.to_string()))?;

        let mut complex = fft
            .inverse(phi_k)
            .await
            .map_err(|e| PppmError::FftError(e.to_string()))?;

        // Normalize for inverse FFT (Fft3DF64 doesn't normalize)
        let scale = 1.0 / (kx * ky * kz) as f64;
        for v in complex.iter_mut() {
            *v *= scale;
        }

        // Extract real part as potential
        let mut values = vec![0.0; size];
        for i in 0..size {
            values[i] = complex[i * 2];
        }

        Ok(PotentialMesh::from_values(
            self.params.mesh_dims,
            values,
            self.params.box_dims,
        ))
    }

    fn backward_fft(&self, phi_k: &[f64]) -> Result<PotentialMesh, PppmError> {
        crate::device::test_pool::tokio_block_on(self.backward_fft_async(phi_k))
    }

    /// CPU 3D FFT using dimension-wise 1D FFTs (test only)
    #[cfg(test)]
    #[expect(dead_code, reason = "CPU reference for GPU validation")]
    fn fft_3d_cpu(
        &self,
        data: &mut [f64],
        nx: usize,
        ny: usize,
        nz: usize,
        inverse: bool,
    ) -> Result<(), PppmError> {
        // FFT along Z (innermost)
        for ix in 0..nx {
            for iy in 0..ny {
                let mut pencil = vec![0.0; nz * 2];
                for iz in 0..nz {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[iz * 2] = data[idx];
                    pencil[iz * 2 + 1] = data[idx + 1];
                }

                self.fft_1d_cpu(&mut pencil, nz, inverse);

                for iz in 0..nz {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[iz * 2];
                    data[idx + 1] = pencil[iz * 2 + 1];
                }
            }
        }

        // FFT along Y
        for ix in 0..nx {
            for iz in 0..nz {
                let mut pencil = vec![0.0; ny * 2];
                for iy in 0..ny {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[iy * 2] = data[idx];
                    pencil[iy * 2 + 1] = data[idx + 1];
                }

                self.fft_1d_cpu(&mut pencil, ny, inverse);

                for iy in 0..ny {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[iy * 2];
                    data[idx + 1] = pencil[iy * 2 + 1];
                }
            }
        }

        // FFT along X (outermost)
        for iy in 0..ny {
            for iz in 0..nz {
                let mut pencil = vec![0.0; nx * 2];
                for ix in 0..nx {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    pencil[ix * 2] = data[idx];
                    pencil[ix * 2 + 1] = data[idx + 1];
                }

                self.fft_1d_cpu(&mut pencil, nx, inverse);

                for ix in 0..nx {
                    let idx = ((ix * ny + iy) * nz + iz) * 2;
                    data[idx] = pencil[ix * 2];
                    data[idx + 1] = pencil[ix * 2 + 1];
                }
            }
        }

        // Normalize for inverse FFT
        if inverse {
            let scale = 1.0 / (nx * ny * nz) as f64;
            for v in data.iter_mut() {
                *v *= scale;
            }
        }

        Ok(())
    }

    /// CPU 1D FFT (Cooley-Tukey radix-2)
    #[cfg(test)]
    fn fft_1d_cpu(&self, data: &mut [f64], n: usize, inverse: bool) {
        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                data.swap(i * 2, j * 2);
                data.swap(i * 2 + 1, j * 2 + 1);
            }
            let mut m = n >> 1;
            while m > 0 && j >= m {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey butterfly
        let sign = if inverse { 1.0 } else { -1.0 };
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let angle_step = sign * 2.0 * PI / len as f64;

            for start in (0..n).step_by(len) {
                let mut angle: f64 = 0.0;
                for k in 0..half {
                    let i = start + k;
                    let j = start + k + half;

                    // Twiddle factor
                    let tw_re = angle.cos();
                    let tw_im = angle.sin();

                    // Butterfly
                    let t_re = tw_re * data[j * 2] - tw_im * data[j * 2 + 1];
                    let t_im = tw_re * data[j * 2 + 1] + tw_im * data[j * 2];

                    data[j * 2] = data[i * 2] - t_re;
                    data[j * 2 + 1] = data[i * 2 + 1] - t_im;
                    data[i * 2] += t_re;
                    data[i * 2 + 1] += t_im;

                    angle += angle_step;
                }
            }
            len *= 2;
        }
    }
}

/// PPPM computation errors
#[derive(Debug, Clone)]
pub enum PppmError {
    /// Position and charge array sizes don't match
    SizeMismatch {
        /// Number of position entries
        positions: usize,
        /// Number of charge entries
        charges: usize,
    },
    /// FFT computation failed
    FftError(String),
}

impl std::fmt::Display for PppmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PppmError::SizeMismatch { positions, charges } => {
                write!(
                    f,
                    "Position count ({positions}) doesn't match charge count ({charges})"
                )
            }
            PppmError::FftError(msg) => write!(f, "FFT error: {msg}"),
        }
    }
}

impl std::error::Error for PppmError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available_sync;
    use crate::ops::md::electrostatics::PppmAccuracy;

    #[test]
    fn test_pppm_two_opposite_charges() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::custom(
            2,
            [10.0, 10.0, 10.0],
            [8, 8, 8],
            2.0, // alpha
            3.0, // rc
            4,   // order
        );

        let pppm = Pppm::new(device, params);

        // Two opposite charges
        let positions = vec![[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]];
        let charges = vec![1.0, -1.0];

        let (forces, energy) = pppm.compute(&positions, &charges).unwrap();

        // Energy should be negative (attractive)
        assert!(energy < 0.0, "Energy should be negative: {}", energy);

        // Forces should attract charges together
        // Positive charge at x=4 should be pulled toward x=6 (positive force)
        assert!(
            forces[0][0] > 0.0,
            "Force on +q should be positive x: {}",
            forces[0][0]
        );
        // Negative charge at x=6 should be pulled toward x=4 (negative force)
        assert!(
            forces[1][0] < 0.0,
            "Force on -q should be negative x: {}",
            forces[1][0]
        );

        // Newton's third law: forces should be equal and opposite
        let force_sum = forces[0][0] + forces[1][0];
        assert!(
            force_sum.abs() < 0.1,
            "Forces should sum to ~0: {}",
            force_sum
        );
    }

    #[test]
    fn test_pppm_two_like_charges() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);

        let pppm = Pppm::new(device, params);

        // Two positive charges
        let positions = vec![[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]];
        let charges = vec![1.0, 1.0];

        let (forces, _energy) = pppm.compute(&positions, &charges).unwrap();

        // Note: Total energy includes negative self-energy correction (-α/√π × Σq²)
        // which can dominate for small systems. The important physical test is
        // that forces are repulsive.

        // Forces should repel: first charge pushed left, second pushed right
        assert!(
            forces[0][0] < 0.0,
            "First charge should be pushed left: {}",
            forces[0][0]
        );
        assert!(
            forces[1][0] > 0.0,
            "Second charge should be pushed right: {}",
            forces[1][0]
        );

        // Newton's third law
        let force_sum = forces[0][0] + forces[1][0];
        assert!(
            force_sum.abs() < 0.1,
            "Forces should sum to ~0: {}",
            force_sum
        );
    }

    #[test]
    fn test_pppm_neutral_system() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::auto(4, 10.0, PppmAccuracy::Medium);
        let pppm = Pppm::new(device, params);

        // Neutral system: +1 +1 -1 -1
        let positions = vec![
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [8.0, 5.0, 5.0],
        ];
        let charges = vec![1.0, 1.0, -1.0, -1.0];

        let (forces, _energy) = pppm.compute(&positions, &charges).unwrap();

        // Total momentum should be conserved (sum of forces = 0)
        let total_fx: f64 = forces.iter().map(|f| f[0]).sum();
        let total_fy: f64 = forces.iter().map(|f| f[1]).sum();
        let total_fz: f64 = forces.iter().map(|f| f[2]).sum();

        assert!(total_fx.abs() < 0.1, "Total Fx should be ~0: {}", total_fx);
        assert!(total_fy.abs() < 0.1, "Total Fy should be ~0: {}", total_fy);
        assert!(total_fz.abs() < 0.1, "Total Fz should be ~0: {}", total_fz);
    }

    #[test]
    fn test_pppm_empty_system() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::auto(0, 10.0, PppmAccuracy::Low);
        let pppm = Pppm::new(device, params);

        let (forces, energy) = pppm.compute(&[], &[]).unwrap();

        assert!(forces.is_empty());
        assert!((energy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pppm_size_mismatch() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::auto(2, 10.0, PppmAccuracy::Low);
        let pppm = Pppm::new(device, params);

        let result = pppm.compute(&[[0.0, 0.0, 0.0]], &[1.0, 2.0]);

        assert!(result.is_err());
        if let Err(PppmError::SizeMismatch { positions, charges }) = result {
            assert_eq!(positions, 1);
            assert_eq!(charges, 2);
        }
    }

    #[test]
    fn test_fft_1d_roundtrip() {
        let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let params = PppmParams::auto(10, 10.0, PppmAccuracy::Medium);
        let pppm = Pppm::new(device, params);

        // Simple complex signal
        let mut data = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0]; // n=4

        let original = data.clone();

        // Forward FFT
        pppm.fft_1d_cpu(&mut data, 4, false);

        // Inverse FFT
        pppm.fft_1d_cpu(&mut data, 4, true);

        // Normalize
        for v in &mut data {
            *v /= 4.0;
        }

        // Should match original
        for (a, b) in original.iter().zip(data.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "FFT roundtrip failed: {} != {}",
                a,
                b
            );
        }
    }
}
