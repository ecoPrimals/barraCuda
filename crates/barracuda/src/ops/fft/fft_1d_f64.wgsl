// 1D Fast Fourier Transform (FFT) - Double Precision (f64) Version
//
// **Purpose**: High-precision frequency domain transformation for PPPM/Ewald
// **Algorithm**: Cooley-Tukey butterfly FFT in complex f64 domain
// **Complexity**: O(n log n) vs O(n²) for naive DFT
//
// **Evolution**: f64 version of fft_1d.wgsl for MD electrostatics
// - Uses scalar f64 arithmetic (no vec2<f64> in WGSL)
// - Uses math_f64.wgsl for sin_f64, cos_f64
// - SAME: Butterfly structure, bit-reversal, stage-wise passes
//
// **Deep Debt Compliance**:
// - ✅ Pure WGSL (no unsafe)
// - ✅ Hardware-agnostic (any wgpu backend with SHADER_F64)
// - ✅ Portable (NVIDIA, AMD, Intel via wgpu + f64 extension)
// - ✅ Numerically precise (IEEE 754 float64)
//
// **CRITICAL**: This shader requires math_f64.wgsl to be prepended!
// Use ShaderTemplate::with_math_f64() in Rust code.

// ============================================================================
// COMPLEX f64 ARITHMETIC
// ============================================================================
// Complex number as pair of f64 (real, imag)
// We use struct since WGSL doesn't have vec2<f64>

struct Complex64 {
    re: f64,
    im: f64,
}

/// Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
fn complex64_add(z1: Complex64, z2: Complex64) -> Complex64 {
    return Complex64(z1.re + z2.re, z1.im + z2.im);
}

/// Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
fn complex64_sub(z1: Complex64, z2: Complex64) -> Complex64 {
    return Complex64(z1.re - z2.re, z1.im - z2.im);
}

/// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
fn complex64_mul(z1: Complex64, z2: Complex64) -> Complex64 {
    let a = z1.re;
    let b = z1.im;
    let c = z2.re;
    let d = z2.im;
    
    // Use FMA for better precision where available
    let re = a * c - b * d;
    let im = a * d + b * c;
    
    return Complex64(re, im);
}

/// Complex exponential: exp(iθ) = cos(θ) + i·sin(θ) (Euler's formula)
/// Uses math_f64.wgsl sin_f64/cos_f64 for double precision
fn complex64_exp_angle(theta: f64) -> Complex64 {
    return Complex64(cos_f64(theta), sin_f64(theta));
}

// ============================================================================
// FFT f64 SHADER
// ============================================================================

// Storage buffers use f64 (stored as pairs: real, imag)
@group(0) @binding(0) var<storage, read> input: array<f64>;           // Complex input (interleaved real, imag)
@group(0) @binding(1) var<storage, read_write> output: array<f64>;    // Complex output
@group(0) @binding(2) var<storage, read> twiddle_re: array<f64>;      // Precomputed twiddle real parts
@group(0) @binding(3) var<storage, read> twiddle_im: array<f64>;      // Precomputed twiddle imag parts
@group(0) @binding(4) var<uniform> params: Fft64Params;

struct Fft64Params {
    degree: u32,      // N (must be power of 2)
    stage: u32,       // Current butterfly stage (0 to log2(N)-1)
    inverse: u32,     // 0 = forward FFT, 1 = inverse FFT
    _padding: u32,    // Alignment
}

/// Load complex number from input buffer at index
fn load_complex_from_input(index: u32) -> Complex64 {
    let base = index * 2u;
    return Complex64(input[base], input[base + 1u]);
}

/// Load complex twiddle factor from separate buffers
fn load_twiddle(index: u32) -> Complex64 {
    return Complex64(twiddle_re[index], twiddle_im[index]);
}

/// Store complex number to output buffer at index
fn store_complex_to_output(index: u32, value: Complex64) {
    let base = index * 2u;
    output[base] = value.re;
    output[base + 1u] = value.im;
}

/// FFT butterfly operation (f64)
///
/// Computes:
///   u = a + twiddle * b
///   v = a - twiddle * b
///
struct Butterfly64Result {
    u: Complex64,
    v: Complex64,
}

fn butterfly64(a: Complex64, b: Complex64, twiddle: Complex64) -> Butterfly64Result {
    let tb = complex64_mul(twiddle, b);  // twiddle * b
    let u = complex64_add(a, tb);        // a + (twiddle * b)
    let v = complex64_sub(a, tb);        // a - (twiddle * b)
    return Butterfly64Result(u, v);
}

/// Bit-reverse permutation helper
fn bit_reverse_index(index: u32, log_n: u32) -> u32 {
    var reversed = 0u;
    var idx = index;
    for (var i = 0u; i < log_n; i = i + 1u) {
        reversed = (reversed << 1u) | (idx & 1u);
        idx = idx >> 1u;
    }
    return reversed;
}

/// Main FFT f64 butterfly kernel
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let butterfly_idx = global_id.x;
    let num_butterflies = params.degree / 2u;
    
    if (butterfly_idx >= num_butterflies) {
        return;
    }
    
    // Butterfly parameters for current stage
    let stage = params.stage;
    let stride = 1u << stage;              // Distance between butterfly pairs
    
    // Compute indices for this butterfly
    let block_size = stride * 2u;
    let block_idx = butterfly_idx / stride;
    let local_idx = butterfly_idx % stride;
    
    let idx_a = block_idx * block_size + local_idx;
    let idx_b = idx_a + stride;
    
    // Load complex operands (f64)
    let a = load_complex_from_input(idx_a);
    let b = load_complex_from_input(idx_b);
    
    // Load twiddle factor (precomputed exp(-2πik/N) for forward FFT).
    // For the inverse FFT we need exp(+2πik/N) = conj(exp(-2πik/N)):
    // negate the imaginary part so the same kernel serves both directions.
    let twiddle_stride = params.degree / (2u * stride);
    let twiddle_idx = local_idx * twiddle_stride;
    var twiddle = load_twiddle(twiddle_idx);
    if params.inverse == 1u {
        twiddle.im = -twiddle.im;
    }

    // Perform butterfly (complex f64 arithmetic)
    let result = butterfly64(a, b, twiddle);
    
    // Store results
    store_complex_to_output(idx_a, result.u);
    store_complex_to_output(idx_b, result.v);
}

/// Bit-reversal permutation kernel (f64)
@compute @workgroup_size(256)
fn bit_reverse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.degree) {
        return;
    }
    
    // Compute bit-reversed index
    let log_n = u32(log2(f32(params.degree)));
    let rev_idx = bit_reverse_index(idx, log_n);
    
    // Only swap if idx < rev_idx (to avoid double-swapping)
    if (idx < rev_idx) {
        let temp = load_complex_from_input(idx);
        let other = load_complex_from_input(rev_idx);
        
        store_complex_to_output(idx, other);
        store_complex_to_output(rev_idx, temp);
    } else if (idx == rev_idx) {
        // Self-symmetric index: just copy
        let value = load_complex_from_input(idx);
        store_complex_to_output(idx, value);
    }
}
