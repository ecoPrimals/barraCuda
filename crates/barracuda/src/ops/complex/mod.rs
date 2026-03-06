// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Number Operations
//!
//! **Purpose**: Complex arithmetic for FFT and wave physics
//!
//! **Architecture**:
//! - Complex numbers represented as `vec2<f32>` (real, imag)
//! - All math in WGSL shaders (universal GPU portability)
//! - All orchestration in Rust (type safety, zero unsafe)
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe code)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (IEEE 754 float32)
//! - ✅ Production-ready (full error handling)
//!
//! ## Operations
//!
//! ### Basic Arithmetic
//! - `complex_add` - Complex addition: (a+bi) + (c+di)
//! - `complex_sub` - Complex subtraction: (a+bi) - (c+di)
//! - `complex_mul` - Complex multiplication: (a+bi)(c+di) ⚠️ **CRITICAL FOR FFT**
//! - `complex_conj` - Complex conjugate: conj(a+bi) = a-bi
//! - `complex_abs` - Magnitude: |a+bi| = sqrt(a²+b²)
//!
//! ### Transcendental Functions
//! - `complex_exp` - Complex exponential: exp(a+bi) ⚠️ **CRITICAL FOR FFT**
//! - `complex_div` - Complex division: (a+bi)/(c+di)
//! - `complex_sqrt` - Complex square root
//! - `complex_log` - Complex logarithm
//! - `complex_pow` - Complex power: z^n
//!
//! ## Usage Example
//!
//! ```no_run
//! use barracuda::ops::complex::ComplexMul;
//! use barracuda::tensor::Tensor;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Arc::new(barracuda::prelude::WgpuDevice::new().await?);
//!
//! // Create complex tensors (interleaved real, imag as f32)
//! // (3+4i), (1+2i)
//! let z1 = Tensor::from_data(&[3.0f32, 4.0, 1.0, 2.0], vec![2, 2], device.clone())?;
//! let z2 = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2], device.clone())?;
//!
//! // Multiply: (3+4i)(1+2i) = -5+10i
//! let op = ComplexMul::new(z1, z2)?;
//! let result = op.execute()?;
//!
//! # Ok(())
//! # }
//! ```
//!
//! ## Design Rationale
//!
//! ### Why `vec2<f32>`?
//! - Native WGSL support (SIMD-friendly)
//! - Direct GPU register mapping
//! - Compatible with texture formats (RG32F)
//! - Minimal overhead vs manual operations
//!
//! ### Why Not Struct?
//! - vec2 gives us native add/sub for free
//! - Better codegen (GPU compilers optimize vec2 paths)
//! - Precedent: GLSL/HLSL use vec2 for complex
//!
//! ### Future: `ComplexF64`
//! - For high-precision physics (energy conservation)
//! - Use `u64_emu` pattern (`vec4<u32>`)
//! - Follows existing FHE precision model
//!
//! ## Mathematical Background
//!
//! Complex numbers: z = a + bi where i² = -1
//!
//! **Euler's Formula**: exp(iθ) = cos(θ) + i·sin(θ)\
//! **De Moivre's Theorem**: (cos(θ) + i·sin(θ))^n = cos(nθ) + i·sin(nθ)\
//! **Conjugate**: conj(z) = a - bi\
//! **Magnitude**: |z| = sqrt(a² + b²)\
//! **Polar Form**: z = r·exp(iθ) where r = |z|, θ = arg(z)

pub mod abs;
pub mod add;
pub mod conj;
pub mod div;
pub mod exp;
pub mod log;
pub mod mul;
pub mod pow;
pub mod sqrt;
pub mod sub;

// Re-export main operations
pub use abs::ComplexAbs;
pub use add::ComplexAdd;
pub use conj::ComplexConj;
pub use div::ComplexDiv;
pub use exp::ComplexExp;
pub use log::ComplexLog;
pub use mul::ComplexMul;
pub use pow::ComplexPow;
pub use sqrt::ComplexSqrt;
pub use sub::ComplexSub;
