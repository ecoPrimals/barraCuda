// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-accelerated 2D Perlin noise and fBm (Fractal Brownian Motion).
//!
//! Computes coherent noise for batch coordinate arrays on the GPU.
//! The standard 256-entry Perlin permutation table is uploaded once as
//! a storage buffer; each thread evaluates noise for one (x, y) pair.
//!
//! # Provenance
//!
//! Absorbed from ludoSpring V2 CPU reference.
//! - Perlin, K. (1985). "An image synthesizer." SIGGRAPH '85.
//! - Perlin, K. (2002). "Improving noise." SIGGRAPH '02.
//! - Gustavson, S. (2005). "Simplex noise demystified."

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const PERLIN_SHADER: &str = include_str!("../../shaders/procedural/perlin_2d_f64.wgsl");
const PERLIN_F32_SHADER: &str = include_str!("../../shaders/procedural/perlin_2d_f32.wgsl");
const FBM_SHADER: &str = include_str!("../../shaders/procedural/fbm_2d_f64.wgsl");

/// Standard Perlin permutation table (Perlin 2002).
const PERM_TABLE: [u32; 256] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180,
];

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PerlinParams {
    n_points: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FbmParams {
    n_points: u32,
    octaves: u32,
    _pad0: u32,
    _pad1: u32,
    lacunarity: f64,
    persistence: f64,
}

/// GPU-accelerated batch 2D Perlin noise.
pub struct PerlinNoiseGpu {
    device: Arc<WgpuDevice>,
}

impl PerlinNoiseGpu {
    /// Create a new Perlin noise compute instance.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute Perlin 2D noise for a batch of (x, y) coordinate pairs.
    ///
    /// `coords` must have even length: `[x0, y0, x1, y1, ...]`.
    /// Returns one f64 per coordinate pair, approximately in \[-1, 1\].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] on GPU dispatch or buffer mapping failure.
    pub fn perlin_2d(&self, coords: &[f64]) -> Result<Vec<f64>> {
        let n_points = coords.len() / 2;
        if n_points == 0 {
            return Ok(vec![]);
        }

        let params = PerlinParams {
            n_points: crate::error::u32_from_usize(n_points)?,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let wg_count = params.n_points.div_ceil(WORKGROUP_SIZE_1D);

        let coord_buf = self.device.create_buffer_f64_init("perlin:coords", coords);
        let out_buf = self.device.create_buffer_f64(n_points)?;
        let perm_buf = self
            .device
            .create_buffer_u32_init("perlin:perm", &PERM_TABLE);
        let params_buf = self.device.create_uniform_buffer("perlin:params", &params);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "perlin_2d")
            .shader(PERLIN_SHADER, "main")
            .f64()
            .storage_read(0, &coord_buf)
            .storage_rw(1, &out_buf)
            .storage_read(2, &perm_buf)
            .uniform(3, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n_points)
    }

    /// Compute fBm (Fractal Brownian Motion) for a batch of (x, y) pairs.
    ///
    /// `coords` must have even length: `[x0, y0, x1, y1, ...]`.
    /// Returns one f64 per coordinate pair, normalized to approximately \[-1, 1\].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] on GPU dispatch or buffer mapping failure.
    pub fn fbm_2d(
        &self,
        coords: &[f64],
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> Result<Vec<f64>> {
        let n_points = coords.len() / 2;
        if n_points == 0 {
            return Ok(vec![]);
        }

        let params = FbmParams {
            n_points: crate::error::u32_from_usize(n_points)?,
            octaves,
            _pad0: 0,
            _pad1: 0,
            lacunarity,
            persistence,
        };

        let wg_count = params.n_points.div_ceil(WORKGROUP_SIZE_1D);

        let coord_buf = self.device.create_buffer_f64_init("fbm:coords", coords);
        let out_buf = self.device.create_buffer_f64(n_points)?;
        let perm_buf = self.device.create_buffer_u32_init("fbm:perm", &PERM_TABLE);
        let params_buf = self.device.create_uniform_buffer("fbm:params", &params);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "fbm_2d")
            .shader(FBM_SHADER, "main")
            .f64()
            .storage_read(0, &coord_buf)
            .storage_rw(1, &out_buf)
            .storage_read(2, &perm_buf)
            .uniform(3, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n_points)
    }
}

/// GPU-accelerated batch 2D Perlin noise (f32 variant).
///
/// Identical algorithm to [`PerlinNoiseGpu`] but operates on f32 data.
/// Does not require the `f64` GPU extension — runs on all WebGPU devices.
/// Designed for game/real-time use (ludoSpring).
pub struct PerlinNoiseGpuF32 {
    device: Arc<WgpuDevice>,
}

impl PerlinNoiseGpuF32 {
    /// Create a new f32 Perlin noise compute instance.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute Perlin 2D noise for a batch of (x, y) coordinate pairs (f32).
    ///
    /// `coords` must have even length: `[x0, y0, x1, y1, ...]`.
    /// Returns one f32 per coordinate pair, approximately in \[-1, 1\].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] on GPU dispatch or buffer mapping failure.
    pub fn perlin_2d(&self, coords: &[f32]) -> Result<Vec<f32>> {
        let n_points = coords.len() / 2;
        if n_points == 0 {
            return Ok(vec![]);
        }

        let params = PerlinParams {
            n_points: crate::error::u32_from_usize(n_points)?,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let wg_count = params.n_points.div_ceil(WORKGROUP_SIZE_1D);

        let coord_buf = self
            .device
            .create_buffer_f32_init("perlin_f32:coords", coords);
        let out_buf = self.device.create_buffer_f32(n_points)?;
        let perm_buf = self
            .device
            .create_buffer_u32_init("perlin_f32:perm", &PERM_TABLE);
        let params_buf = self
            .device
            .create_uniform_buffer("perlin_f32:params", &params);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "perlin_2d_f32")
            .shader(PERLIN_F32_SHADER, "main")
            .storage_read(0, &coord_buf)
            .storage_rw(1, &out_buf)
            .storage_read(2, &perm_buf)
            .uniform(3, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_buffer_f32(&out_buf, n_points)
    }
}

/// Floor-and-mask to permutation table index (0..=255).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "floor + mask to 0..255 guarantees valid usize index"
)]
fn perm_index(v: f64) -> usize {
    v.floor() as usize & 255
}

/// f32 variant of [`perm_index`].
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "floor + mask to 0..255 guarantees valid usize index"
)]
fn perm_index_f32(v: f32) -> usize {
    v.floor() as usize & 255
}

/// CPU reference: 2D Perlin noise (f32). Returns value in approximately \[-1, 1\].
///
/// Matches the f32 GPU shader output for cross-validation.
#[must_use]
pub fn perlin_2d_cpu_f32(x: f32, y: f32) -> f32 {
    fn fade(t: f32) -> f32 {
        t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
    }
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        t.mul_add(b - a, a)
    }
    fn grad2(hash: u32, x: f32, y: f32) -> f32 {
        match hash & 3 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            _ => -x - y,
        }
    }

    let xi = perm_index_f32(x);
    let yi = perm_index_f32(y);
    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = PERM_TABLE[(PERM_TABLE[xi] as usize + yi) & 255];
    let ab = PERM_TABLE[(PERM_TABLE[xi] as usize + yi + 1) & 255];
    let ba = PERM_TABLE[(PERM_TABLE[(xi + 1) & 255] as usize + yi) & 255];
    let bb = PERM_TABLE[(PERM_TABLE[(xi + 1) & 255] as usize + yi + 1) & 255];

    lerp(
        lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u),
        lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u),
        v,
    )
}

/// CPU reference: 2D Perlin noise. Returns value in approximately \[-1, 1\].
///
/// Matches the GPU shader output for cross-validation.
#[must_use]
pub fn perlin_2d_cpu(x: f64, y: f64) -> f64 {
    fn fade(t: f64) -> f64 {
        t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
    }
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        t.mul_add(b - a, a)
    }
    fn grad2(hash: u32, x: f64, y: f64) -> f64 {
        match hash & 3 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            _ => -x - y,
        }
    }

    let xi = perm_index(x);
    let yi = perm_index(y);
    let xf = x - x.floor();
    let yf = y - y.floor();

    let u = fade(xf);
    let v = fade(yf);

    let aa = PERM_TABLE[(PERM_TABLE[xi] as usize + yi) & 255];
    let ab = PERM_TABLE[(PERM_TABLE[xi] as usize + yi + 1) & 255];
    let ba = PERM_TABLE[(PERM_TABLE[(xi + 1) & 255] as usize + yi) & 255];
    let bb = PERM_TABLE[(PERM_TABLE[(xi + 1) & 255] as usize + yi + 1) & 255];

    lerp(
        lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u),
        lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u),
        v,
    )
}

/// CPU reference: 3D Perlin noise. Returns value in approximately \[-1, 1\].
///
/// Classic Perlin (2002) with 3D gradient vectors, trilinear interpolation,
/// and quintic fade. Guarantees zero at all integer lattice points.
///
/// # References
///
/// - Perlin, K. (2002). "Improving noise." SIGGRAPH '02.
#[must_use]
pub fn perlin_3d_cpu(x: f64, y: f64, z: f64) -> f64 {
    fn fade(t: f64) -> f64 {
        t * t * t * t.mul_add(t.mul_add(6.0, -15.0), 10.0)
    }
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        t.mul_add(b - a, a)
    }
    fn grad3(hash: u32, x: f64, y: f64, z: f64) -> f64 {
        match hash & 15 {
            0 => x + y,
            1 => -x + y,
            2 => x - y,
            3 => -x - y,
            4 => x + z,
            5 => -x + z,
            6 => x - z,
            7 => -x - z,
            8 => y + z,
            9 => -y + z,
            10 => y - z,
            11 => -y - z,
            12 => y + x,
            13 => -y + z,
            14 => y - x,
            _ => -y - z,
        }
    }

    let xi = perm_index(x);
    let yi = perm_index(y);
    let zi = perm_index(z);
    let xf = x - x.floor();
    let yf = y - y.floor();
    let zf = z - z.floor();

    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);

    let p = &PERM_TABLE;
    let a = p[xi] as usize + yi;
    let aa = p[a & 255] as usize + zi;
    let ab = p[(a + 1) & 255] as usize + zi;
    let b = p[(xi + 1) & 255] as usize + yi;
    let ba = p[b & 255] as usize + zi;
    let bb = p[(b + 1) & 255] as usize + zi;

    lerp(
        lerp(
            lerp(
                grad3(p[aa & 255], xf, yf, zf),
                grad3(p[ba & 255], xf - 1.0, yf, zf),
                u,
            ),
            lerp(
                grad3(p[ab & 255], xf, yf - 1.0, zf),
                grad3(p[bb & 255], xf - 1.0, yf - 1.0, zf),
                u,
            ),
            v,
        ),
        lerp(
            lerp(
                grad3(p[(aa + 1) & 255], xf, yf, zf - 1.0),
                grad3(p[(ba + 1) & 255], xf - 1.0, yf, zf - 1.0),
                u,
            ),
            lerp(
                grad3(p[(ab + 1) & 255], xf, yf - 1.0, zf - 1.0),
                grad3(p[(bb + 1) & 255], xf - 1.0, yf - 1.0, zf - 1.0),
                u,
            ),
            v,
        ),
        w,
    )
}

/// CPU reference: 2D fBm using Perlin noise.
#[must_use]
pub fn fbm_2d_cpu(x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_value = 0.0;

    for _ in 0..octaves {
        value += perlin_2d_cpu(x * frequency, y * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    value / max_value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_2d_cpu_in_range() {
        for i in 0..50_i32 {
            for j in 0..50_i32 {
                let v = perlin_2d_cpu(f64::from(i) * 0.1, f64::from(j) * 0.1);
                assert!((-2.0..=2.0).contains(&v), "value {v} out of range");
            }
        }
    }

    #[test]
    fn perlin_at_integer_coords_is_zero() {
        let v = perlin_2d_cpu(1.0, 1.0);
        assert!(
            v.abs() < 1e-10,
            "perlin at integer coords should be ~0, got {v}"
        );
    }

    #[test]
    fn fbm_is_coherent() {
        let a = fbm_2d_cpu(1.0, 1.0, 4, 2.0, 0.5);
        let b = fbm_2d_cpu(1.001, 1.001, 4, 2.0, 0.5);
        assert!((a - b).abs() < 0.1, "nearby samples should be similar");
    }

    #[test]
    fn fbm_normalizes_output() {
        let v = fbm_2d_cpu(0.5, 0.5, 8, 2.0, 0.5);
        assert!(
            (-2.0..=2.0).contains(&v),
            "fbm output {v} should be bounded"
        );
    }

    #[test]
    fn params_layout_perlin() {
        assert_eq!(std::mem::size_of::<PerlinParams>(), 16);
    }

    #[test]
    fn params_layout_fbm() {
        assert_eq!(std::mem::size_of::<FbmParams>(), 32);
    }

    #[test]
    fn shader_source_valid() {
        assert!(PERLIN_SHADER.contains("perlin"));
        assert!(PERLIN_SHADER.contains("perm_lookup"));
        assert!(FBM_SHADER.contains("octaves"));
        assert!(FBM_SHADER.contains("perlin_2d"));
    }

    #[test]
    fn perm_table_contains_all_values() {
        let mut seen = [false; 256];
        for &v in &PERM_TABLE {
            assert!(v < 256, "perm table value {v} >= 256");
            seen[v as usize] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "perm table missing value {i}");
        }
    }

    #[test]
    fn perlin_3d_cpu_in_range() {
        for i in 0..10_i32 {
            for j in 0..10_i32 {
                for k in 0..10_i32 {
                    let v =
                        perlin_3d_cpu(f64::from(i) * 0.3, f64::from(j) * 0.3, f64::from(k) * 0.3);
                    assert!((-2.0..=2.0).contains(&v), "3D value {v} out of range");
                }
            }
        }
    }

    #[test]
    fn perlin_3d_at_integer_coords_is_zero() {
        for x in 0..5_i32 {
            for y in 0..5_i32 {
                for z in 0..5_i32 {
                    let v = perlin_3d_cpu(f64::from(x), f64::from(y), f64::from(z));
                    assert!(
                        v.abs() < 1e-10,
                        "3D perlin at integer ({x},{y},{z}) should be ~0, got {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn perlin_3d_is_coherent() {
        let a = perlin_3d_cpu(0.5, 0.5, 0.5);
        let b = perlin_3d_cpu(0.501, 0.501, 0.501);
        assert!((a - b).abs() < 0.1, "nearby 3D samples should be similar");
    }

    #[test]
    fn perlin_2d_f32_in_range() {
        for i in 0..50_i32 {
            for j in 0..50_i32 {
                let v = perlin_2d_cpu_f32(i as f32 * 0.1, j as f32 * 0.1);
                assert!((-2.0..=2.0).contains(&v), "f32 value {v} out of range");
            }
        }
    }

    #[test]
    fn perlin_2d_f32_at_integer_is_zero() {
        let v = perlin_2d_cpu_f32(1.0, 1.0);
        assert!(
            v.abs() < 1e-5,
            "f32 perlin at integer coords should be ~0, got {v}"
        );
    }

    #[test]
    fn perlin_2d_f32_matches_f64() {
        for i in 0..20_i32 {
            let x = (i as f64).mul_add(0.37, 0.1);
            let y = (i as f64).mul_add(0.41, 0.2);
            let v64 = perlin_2d_cpu(x, y);
            let v32 = perlin_2d_cpu_f32(x as f32, y as f32);
            assert!(
                (v64 as f32 - v32).abs() < 0.01,
                "f32/f64 mismatch at ({x},{y}): f64={v64}, f32={v32}"
            );
        }
    }

    #[test]
    fn f32_shader_source_valid() {
        assert!(PERLIN_F32_SHADER.contains("perlin"));
        assert!(PERLIN_F32_SHADER.contains("perm_lookup"));
        assert!(
            !PERLIN_F32_SHADER.contains("enable f64;"),
            "f32 shader must not use the f64 WGSL extension"
        );
    }
}
