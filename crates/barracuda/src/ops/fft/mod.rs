// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fast Fourier Transform (FFT) Operations
//!
//! **Purpose**: Complex FFT for wave physics and frequency analysis
//!
//! **Architecture**: GPU-resident f64-canonical
//! - All FFT shaders written in f64 (fft_3d_batched_f64.wgsl)
//! - f32 path: downcast via `downcast_f64_to_f32` for broad compatibility
//! - f64 path: native when device has `SHADER_F64`, else DF64 fallback
//! - Zero CPU readbacks for f32 pipeline (Fft1D, Fft2D, Fft3D, Rfft)
//!
//! **Evolution**: Adapted from FHE NTT (80% structure reuse!)
//! - NTT: Butterfly in integer modular domain
//! - FFT: Butterfly in complex float domain
//! - SAME algorithm (Cooley-Tukey), DIFFERENT arithmetic
//!
//! ## Operations
//!
//! - `fft_1d` - 1D Complex FFT (Cooley-Tukey radix-2, f32) ⚠️ **CRITICAL FOR PPPM**
//! - `fft_1d_f64` - 1D Complex FFT (f64) for high-precision MD simulations
//! - `ifft_1d` - 1D Inverse FFT (conjugate + normalize)
//! - `fft_2d` - 2D FFT via row-column decomposition
//! - `fft_3d` - 3D FFT for PPPM long-range forces
//! - `rfft` - Real-to-complex FFT (optimized)
//!
//! ## Mathematical Background
//!
//! The Discrete Fourier Transform (DFT):
//! ```text
//! X[k] = Σ(n=0 to N-1) x[n] · exp(-2πikn/N)
//! ```
//!
//! FFT accelerates this from O(N²) to O(N log N) via Cooley-Tukey butterfly.
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use barracuda::ops::fft::Fft1D;
//! use barracuda::prelude::{WgpuDevice, Tensor};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Arc::new(WgpuDevice::new().await?);
//!
//! let signal = Tensor::from_data(&signal_data, vec![1024, 2], device.clone())?;
//! let fft = Fft1D::new(signal, 1024)?;
//! let spectrum = fft.execute()?;
//! # Ok(())
//! # }
//! ```

pub mod fft_1d;
pub mod fft_1d_f64;
pub mod fft_2d;
pub mod fft_3d;
pub mod fft_3d_f64;
pub mod ifft_1d;
pub mod rfft;

#[cfg(test)]
mod tests;

pub use fft_1d::Fft1D;
pub use fft_1d_f64::Fft1DF64;
pub use fft_2d::Fft2D;
pub use fft_3d::Fft3D;
pub use fft_3d_f64::Fft3DF64;
pub use ifft_1d::Ifft1D;
pub use rfft::Rfft;
