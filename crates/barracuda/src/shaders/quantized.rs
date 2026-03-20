// SPDX-License-Identifier: AGPL-3.0-or-later
//! Quantized Inference Shaders
//!
//! WGSL shaders for efficient inference with quantized weights.
//!
//! ## Supported Quantization Types
//!
//! - **`Q4_0`**: 4-bit quantization (llama.cpp format)
//!   - 32 elements per block
//!   - 18 bytes per block (2 scale + 16 data)
//!   - ~4.5x memory reduction vs f32
//!
//! - **`Q8_0`**: 8-bit quantization
//!   - 32 elements per block
//!   - 34 bytes per block (2 scale + 32 data)
//!   - ~4x memory reduction vs f32
//!
//! ## Performance Notes
//!
//! On-the-fly dequantization during GEMV is often faster than:
//! 1. Dequantizing all weights to f32
//! 2. Then performing f32 GEMM
//!
//! Because:
//! - Memory bandwidth is the bottleneck for LLM inference
//! - Quantized weights = less data to read from VRAM
//! - GPU compute can hide dequantization latency

/// `Q4_0` dequantization shader source (f64 canonical, downcast to f32)
#[must_use]
pub fn dequant_q4_wgsl() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("quantized/dequant_q4_f64.wgsl").to_string());
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// `Q8_0` dequantization shader source (f64 canonical, downcast to f32)
#[must_use]
pub fn dequant_q8_wgsl() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("quantized/dequant_q8_f64.wgsl").to_string());
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// `Q4_0` GEMV shader source (on-the-fly dequantization, f64 canonical)
#[must_use]
pub fn gemv_q4_wgsl() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("quantized/gemv_q4_f64.wgsl").to_string());
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// `Q8_0` GEMV shader source (on-the-fly dequantization, f64 canonical)
#[must_use]
pub fn gemv_q8_wgsl() -> &'static str {
    static SHADER: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(|| include_str!("quantized/gemv_q8_f64.wgsl").to_string());
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// Block size for `Q4_0` and `Q8_0` quantization
pub const QUANT_BLOCK_SIZE: usize = 32;

/// Bytes per `Q4_0` block
pub const Q4_BYTES_PER_BLOCK: usize = 18;

/// Bytes per `Q8_0` block
pub const Q8_BYTES_PER_BLOCK: usize = 34;

/// Quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    /// 4-bit quantization (llama.cpp `Q4_0`)
    Q4_0,
    /// 8-bit quantization (llama.cpp `Q8_0`)
    Q8_0,
}

impl QuantType {
    /// Get block size for this quantization type
    #[must_use]
    pub const fn block_size(&self) -> usize {
        QUANT_BLOCK_SIZE
    }

    /// Get bytes per block for this quantization type
    #[must_use]
    pub const fn bytes_per_block(&self) -> usize {
        match self {
            Self::Q4_0 => Q4_BYTES_PER_BLOCK,
            Self::Q8_0 => Q8_BYTES_PER_BLOCK,
        }
    }

    /// Calculate total bytes needed for `numel` elements
    #[must_use]
    pub fn required_bytes(&self, numel: usize) -> usize {
        let num_blocks = numel.div_ceil(self.block_size());
        num_blocks * self.bytes_per_block()
    }

    /// Memory reduction ratio compared to f32
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = QUANT_BLOCK_SIZE * 4;
        let quant_bytes = self.bytes_per_block();
        f32_bytes as f32 / quant_bytes as f32
    }

    /// Get dequantization shader source
    #[must_use]
    pub fn dequant_shader(&self) -> &'static str {
        match self {
            Self::Q4_0 => dequant_q4_wgsl(),
            Self::Q8_0 => dequant_q8_wgsl(),
        }
    }

    /// Get GEMV shader source
    #[must_use]
    pub fn gemv_shader(&self) -> &'static str {
        match self {
            Self::Q4_0 => gemv_q4_wgsl(),
            Self::Q8_0 => gemv_q8_wgsl(),
        }
    }
}

/// CPU reference implementation for `Q4_0` dequantization
#[must_use]
pub fn dequant_q4_cpu(data: &[u8], numel: usize) -> Vec<f32> {
    let block_size = QUANT_BLOCK_SIZE;
    let n_blocks = numel.div_ceil(block_size);
    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * Q4_BYTES_PER_BLOCK;
        if block_offset + Q4_BYTES_PER_BLOCK > data.len() {
            break;
        }

        // Read scale (f16)
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read quantized values (4 bits each)
        for i in 0..16 {
            if output.len() >= numel {
                break;
            }
            let byte = data[block_offset + 2 + i];
            let q0 = (byte & 0x0F) as i8 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;

            output.push(scale * q0 as f32);
            if output.len() < numel {
                output.push(scale * q1 as f32);
            }
        }
    }

    output.truncate(numel);
    output
}

/// CPU reference implementation for `Q8_0` dequantization
#[must_use]
pub fn dequant_q8_cpu(data: &[u8], numel: usize) -> Vec<f32> {
    let block_size = QUANT_BLOCK_SIZE;
    let n_blocks = numel.div_ceil(block_size);
    let mut output = Vec::with_capacity(numel);

    for block_idx in 0..n_blocks {
        let block_offset = block_idx * Q8_BYTES_PER_BLOCK;
        if block_offset + Q8_BYTES_PER_BLOCK > data.len() {
            break;
        }

        // Read scale (f16)
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = half::f16::from_le_bytes(scale_bytes).to_f32();

        // Read quantized values (8 bits each, signed)
        for i in 0..32 {
            if output.len() >= numel {
                break;
            }
            let q = data[block_offset + 2 + i] as i8;
            output.push(scale * q as f32);
        }
    }

    output.truncate(numel);
    output
}

/// CPU reference implementation for quantized GEMV
#[must_use]
pub fn gemv_quantized_cpu(
    a_quant: &[u8],
    x: &[f32],
    m: usize,
    k: usize,
    quant_type: QuantType,
) -> Vec<f32> {
    let mut y = vec![0.0f32; m];
    let k_blocks = k.div_ceil(quant_type.block_size());
    let bytes_per_block = quant_type.bytes_per_block();

    for row in 0..m {
        let row_offset = row * k_blocks * bytes_per_block;
        let row_data = &a_quant[row_offset..row_offset + k_blocks * bytes_per_block];

        // Dequantize row and compute dot product
        let row_f32 = match quant_type {
            QuantType::Q4_0 => dequant_q4_cpu(row_data, k),
            QuantType::Q8_0 => dequant_q8_cpu(row_data, k),
        };

        y[row] = row_f32.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_type_properties() {
        assert_eq!(QuantType::Q4_0.block_size(), 32);
        assert_eq!(QuantType::Q4_0.bytes_per_block(), 18);
        assert_eq!(QuantType::Q8_0.bytes_per_block(), 34);

        // Q4 has ~7x compression
        assert!(QuantType::Q4_0.compression_ratio() > 7.0);
        // Q8 has ~3.8x compression
        assert!(QuantType::Q8_0.compression_ratio() > 3.5);
    }

    #[test]
    fn test_required_bytes() {
        // 64 elements = 2 blocks
        assert_eq!(QuantType::Q4_0.required_bytes(64), 36);
        assert_eq!(QuantType::Q8_0.required_bytes(64), 68);

        // 33 elements = 2 blocks (rounded up)
        assert_eq!(QuantType::Q4_0.required_bytes(33), 36);
    }

    #[test]
    fn test_dequant_q8_cpu() {
        // Create a simple Q8 block with scale=1.0
        let mut data = vec![0u8; 34];
        // Scale = 1.0 as f16 = 0x3C00
        data[0] = 0x00;
        data[1] = 0x3C;
        // Set quantized values
        data[2] = 10; // 10
        data[3] = 246; // -10 (as u8)

        let result = dequant_q8_cpu(&data, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 10.0).abs() < 0.1);
        assert!((result[1] - (-10.0)).abs() < 0.1);
    }

    #[test]
    fn test_shader_sources_exist() {
        assert!(!dequant_q4_wgsl().is_empty());
        assert!(!dequant_q8_wgsl().is_empty());
        assert!(!gemv_q4_wgsl().is_empty());
        assert!(!gemv_q8_wgsl().is_empty());
    }
}
