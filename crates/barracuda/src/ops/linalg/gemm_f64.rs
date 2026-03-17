// SPDX-License-Identifier: AGPL-3.0-only
//! Dense Matrix Multiply (f64) — GPU-Accelerated via WGSL
//!
//! Batched GEMM: C = alpha * A * B + beta * C
//! Supports batched, matrix-vector, and element-wise operations.
//!
//! **Use cases**:
//! - HFB Hamiltonian assembly (radial integrals as matrix products)
//! - Density computation (matrix-vector products)
//! - Energy functional evaluation
//! - Any dense f64 linear algebra on GPU
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Safe Rust wrapper (no unsafe code)
//! - Runtime-configured dimensions and batch size

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const WGSL_DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
const GEMM_SHADER_DF64: &str = include_str!("../../shaders/linalg/gemm_df64.wgsl");

/// Parameters for GEMM shader (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GemmParams {
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
    alpha_lo: u32, // f64 split into two u32s for Pod
    alpha_hi: u32,
    beta_lo: u32,
    beta_hi: u32,
    trans_a: u32, // 0 = no transpose, 1 = transpose A
    trans_b: u32, // 0 = no transpose, 1 = transpose B
    _pad0: u32,
    _pad1: u32,
}

impl GemmParams {
    fn new(
        m: u32,
        k: u32,
        n: u32,
        batch_size: u32,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) -> Self {
        let alpha_bits = alpha.to_bits();
        let beta_bits = beta.to_bits();
        GemmParams {
            m,
            k,
            n,
            batch_size,
            alpha_lo: alpha_bits as u32,
            alpha_hi: (alpha_bits >> 32) as u32,
            beta_lo: beta_bits as u32,
            beta_hi: (beta_bits >> 32) as u32,
            trans_a: u32::from(trans_a),
            trans_b: u32::from(trans_b),
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// GPU-accelerated dense matrix multiply (f64)
pub struct GemmF64;

impl GemmF64 {
    /// The raw WGSL source for the GEMM f64 shader.
    /// Exposed so downstream crates can include this source in their own fused pipelines
    /// without fragile cross-crate `include_str!` paths.
    pub const WGSL: &'static str = include_str!("../../shaders/linalg/gemm_f64.wgsl");

    fn wgsl_shader() -> &'static str {
        Self::WGSL
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let profile = GpuDriverProfile::from_device(device);
        let strategy = profile.fp64_strategy();
        tracing::info!(?strategy, "GEMM: using {:?} FP64 strategy", strategy);
        match strategy {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => format!("{WGSL_DF64_CORE}\n{GEMM_SHADER_DF64}"),
        }
    }

    /// Execute batched matrix multiply: C = A * B
    /// # Arguments
    /// * `device` - `WgpuDevice`
    /// * `a` - Packed A matrices [`batch_size` × M × K] row-major f64
    /// * `b` - Packed B matrices [`batch_size` × K × N] row-major f64
    /// * `m` - Rows of A / C
    /// * `k` - Cols of A / Rows of B
    /// * `n` - Cols of B / C
    /// * `batch_size` - Number of independent multiplications
    /// # Returns
    /// C matrices [`batch_size` × M × N] row-major f64
    /// # Errors
    /// Returns [`Err`] if [`execute_gemm`](Self::execute_gemm) fails (invalid dimensions or
    /// buffer readback failure).
    pub fn execute(
        device: Arc<WgpuDevice>,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<f64>> {
        Self::execute_gemm(device, a, b, m, k, n, batch_size, 1.0, 0.0)
    }

    /// Execute batched GEMM with alpha/beta: C = alpha * A * B + beta * C
    /// # Errors
    /// Returns [`Err`] if `a.len() != batch_size * m * k` or `b.len() != batch_size * k * n`
    /// (invalid dimensions), or if buffer allocation or readback fails (e.g., device lost).
    pub fn execute_gemm(
        device: Arc<WgpuDevice>,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Vec<f64>> {
        Self::execute_gemm_ex(device, a, b, m, k, n, batch_size, alpha, beta, false, false)
    }

    /// Execute batched GEMM with transpose flags:
    /// `C = alpha * op(A) * op(B) + beta * C`
    /// where `op(X) = X` if `trans_x == false`, `op(X) = X^T` if `trans_x == true`.
    ///
    /// Storage layout follows the transpose convention:
    /// - `trans_a == false`: A is `[batch × M × K]` row-major
    /// - `trans_a == true`:  A is `[batch × K × M]` row-major (logically transposed to `[M × K]`)
    /// - `trans_b == false`: B is `[batch × K × N]` row-major
    /// - `trans_b == true`:  B is `[batch × N × K]` row-major (logically transposed to `[K × N]`)
    ///
    /// This enables `A^T * A` and `A^T * b` patterns without materializing the
    /// transpose — critical for Tikhonov regularization and least-squares solvers.
    /// # Errors
    /// Returns [`Err`] if dimensions don't match storage layout, or if buffer
    /// allocation or readback fails (e.g., device lost).
    pub fn execute_gemm_ex(
        device: Arc<WgpuDevice>,
        a: &[f64],
        b: &[f64],
        m: usize,
        k: usize,
        n: usize,
        batch_size: usize,
        alpha: f64,
        beta: f64,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Vec<f64>> {
        // Storage sizes account for transpose: A is [K×M] when transposed
        let expected_a = batch_size * m * k;
        let expected_b = batch_size * k * n;
        if a.len() != expected_a {
            return Err(BarracudaError::InvalidInput {
                message: format!("A: expected {} elements, got {}", expected_a, a.len()),
            });
        }
        if b.len() != expected_b {
            return Err(BarracudaError::InvalidInput {
                message: format!("B: expected {} elements, got {}", expected_b, b.len()),
            });
        }

        let c_size = batch_size * m * n;

        let a_bytes: &[u8] = bytemuck::cast_slice(a);
        let a_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GEMM A f64"),
                contents: a_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_bytes: &[u8] = bytemuck::cast_slice(b);
        let b_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GEMM B f64"),
                contents: b_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let c_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GEMM C f64"),
            size: (c_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GemmParams::new(
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
            alpha,
            beta,
            trans_a,
            trans_b,
        );
        let params_buffer = device.create_uniform_buffer("GEMM Params", &params);

        use crate::device::compute_pipeline::ComputeDispatch;

        let src = Self::wgsl_shader_for_device(&device);

        let wg_x = (n as u32).div_ceil(16);
        let wg_y = (m as u32).div_ceil(16);
        let wg_z = batch_size as u32;

        ComputeDispatch::new(&device, "gemm_f64")
            .shader(&src, "gemm_f64")
            .f64()
            .uniform(0, &params_buffer)
            .storage_read(1, &a_buffer)
            .storage_read(2, &b_buffer)
            .storage_rw(3, &c_buffer)
            .dispatch(wg_x, wg_y, wg_z)
            .submit()?;

        device.read_f64_buffer(&c_buffer, c_size)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GemmCachedF64 — Pre-compiled GEMM with resident weight matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-compiled GEMM with a GPU-resident weight matrix B.
///
/// Absorbed from wetSpring's `GemmCached` local extension.  Key insight:
/// in taxonomy / ML inference pipelines, the weight matrix B is constant
/// across thousands of samples.  Uploading it once and reusing it collapses
/// per-sample overhead from ~2 ms (upload + pipeline compile) to <0.2 ms.
///
/// **Measured wetSpring improvement**: 60× speedup on taxonomy dispatch
/// (first dispatch: 60 ms → subsequent: <1 ms).
///
/// # Example
/// ```ignore
/// // Pre-compile once at startup with constant weight matrix
/// let gemm = GemmCachedF64::new(device, &weights, k, n, batch)?;
/// // For each sample:
/// let output = gemm.multiply(&sample_features, m)?;
/// ```
pub struct GemmCachedF64 {
    device: Arc<WgpuDevice>,
    pipeline: Arc<wgpu::ComputePipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
    b_buffer: Arc<wgpu::Buffer>,
    /// Cols of B / Rows of A (K dimension)
    pub k: usize,
    /// Cols of C / Cols of B (N dimension)
    pub n: usize,
    /// Batch size
    pub batch_size: usize,
}

impl GemmCachedF64 {
    /// Pre-compile the GEMM pipeline and upload weight matrix B to GPU.
    /// # Arguments
    /// * `device`     — GPU device
    /// * `b`          — Weight matrix `[batch × K × N]` row-major f64
    /// * `k`          — Inner dimension
    /// * `n`          — Output columns
    /// * `batch_size` — Number of independent GEMM operations
    /// # Errors
    /// Returns [`Err`] if `b.len() != batch_size * k * n` (invalid dimensions), or if buffer
    /// allocation or pipeline compilation fails (e.g., device lost, out of memory).
    pub fn new(
        device: Arc<WgpuDevice>,
        b: &[f64],
        k: usize,
        n: usize,
        batch_size: usize,
    ) -> Result<Self> {
        if b.len() != batch_size * k * n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "GemmCachedF64: B has {} elements, expected batch×K×N={}",
                    b.len(),
                    batch_size * k * n
                ),
            });
        }

        let dev = &device;

        // Upload B to GPU permanently — it stays resident across all multiply() calls.
        let b_buffer = Arc::new(
            dev.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("GemmCached B (weight matrix)"),
                    contents: bytemuck::cast_slice(b),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );

        // Create shared bind group layout (reused for every bind group).
        let bgl = Arc::new(
            dev.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GemmCached BGL"),
                    entries: &[
                        bgl_storage_ro(0),
                        bgl_storage_ro(1),
                        bgl_storage_rw(2),
                        bgl_uniform(3),
                    ],
                }),
        );

        // Compile the pipeline exactly once.
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GemmCached PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let src = GemmF64::wgsl_shader_for_device(dev);
        let shader = dev.compile_shader_f64(&src, Some("GemmCached"));
        let pipeline = Arc::new(dev.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("GemmCached Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("gemm_f64"),
                cache: None,
                compilation_options: Default::default(),
            },
        ));

        Ok(Self {
            device,
            pipeline,
            bgl,
            b_buffer,
            k,
            n,
            batch_size,
        })
    }

    /// Multiply input matrix A by the pre-loaded weight matrix B.
    /// # Arguments
    /// * `a` — `[batch × M × K]` row-major f64 input
    /// * `m` — Rows of A (varies per call; K must match the B dimensions)
    /// # Returns
    /// `[batch × M × N]` row-major f64 output (read back to CPU).
    /// # Errors
    /// Returns [`Err`] if [`execute_to_buffer`](Self::execute_to_buffer) fails (invalid dimensions),
    /// or if buffer readback fails (e.g., device lost).
    pub fn multiply(&self, a: &[f64], m: usize) -> Result<Vec<f64>> {
        let c_buf = self.execute_to_buffer(a, m)?;
        let c_size = self.batch_size * m * self.n;
        self.device.read_f64_buffer(&c_buf, c_size)
    }

    /// GPU-resident GEMM — returns `wgpu::Buffer` without CPU readback.
    /// Use this in streaming pipelines where the output feeds directly
    /// into another GPU dispatch (e.g., HFB self-consistency, density mixing).
    /// # Returns
    /// GPU buffer containing `[batch × M × N]` row-major f64 output.
    /// # Errors
    /// Returns [`Err`] if `a.len() != batch_size * m * k` (invalid dimensions), or if buffer
    /// allocation or GPU dispatch fails (e.g., device lost, out of memory).
    pub fn execute_to_buffer(&self, a: &[f64], m: usize) -> Result<wgpu::Buffer> {
        let k = self.k;
        let n = self.n;
        let batch_size = self.batch_size;

        if a.len() != batch_size * m * k {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "GemmCachedF64::execute_to_buffer: A has {} elements, expected batch×M×K={}",
                    a.len(),
                    batch_size * m * k
                ),
            });
        }

        let dev = &self.device;
        let c_size = batch_size * m * n;

        let params = GemmParams::new(
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
            1.0,
            0.0,
            false,
            false,
        );
        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GemmCached Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let a_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GemmCached A"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let c_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GemmCached C"),
            size: (c_size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GemmCached BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let tile = 16u32;
        let wg_x = (m as u32).div_ceil(tile);
        let wg_y = (n as u32).div_ceil(tile);
        let wg_z = batch_size as u32;

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GemmCached Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GemmCached Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }
        dev.submit_commands(Some(encoder.finish()));

        Ok(c_buf)
    }
}

// ── bind group layout helpers ─────────────────────────────────────────────────

fn bgl_storage_ro(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMM_DF64_SHADER: &str = include_str!("../../shaders/linalg/gemm_df64.wgsl");

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn gemm_params_layout() {
        assert_eq!(std::mem::size_of::<GemmParams>(), 48);
    }

    #[test]
    fn gemm_f64_shader_source_valid() {
        let source = GemmF64::WGSL;
        assert!(!source.is_empty());
        assert!(source.contains("fn main") || source.contains("@compute"));
    }

    #[test]
    fn gemm_df64_shader_source_valid() {
        assert!(!GEMM_DF64_SHADER.is_empty());
        assert!(GEMM_DF64_SHADER.contains("fn main") || GEMM_DF64_SHADER.contains("@compute"));
    }

    #[tokio::test]
    async fn test_gemm_2x2() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];

        let c = GemmF64::execute(device, &a, &b, 2, 2, 2, 1).unwrap();

        assert_eq!(c.len(), 4);
        assert!(approx_eq(c[0], 19.0, 1e-10));
        assert!(approx_eq(c[1], 22.0, 1e-10));
        assert!(approx_eq(c[2], 43.0, 1e-10));
        assert!(approx_eq(c[3], 50.0, 1e-10));
    }

    #[tokio::test]
    async fn test_gemm_transpose_a() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // A stored as [K×M] = [[1, 3], [2, 4]] (column-major of [[1,2],[3,4]])
        // trans_a=true means read as A^T = [[1,2],[3,4]] (M=2, K=2)
        // B = [[5, 6], [7, 8]]
        // A^T * B = [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a_stored = vec![1.0_f64, 3.0, 2.0, 4.0]; // [K=2 × M=2]
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];

        let c = GemmF64::execute_gemm_ex(device, &a_stored, &b, 2, 2, 2, 1, 1.0, 0.0, true, false)
            .unwrap();

        assert_eq!(c.len(), 4);
        assert!(approx_eq(c[0], 19.0, 1e-10));
        assert!(approx_eq(c[1], 22.0, 1e-10));
        assert!(approx_eq(c[2], 43.0, 1e-10));
        assert!(approx_eq(c[3], 50.0, 1e-10));
    }

    #[tokio::test]
    async fn test_gemm_transpose_b() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // A = [[1, 2], [3, 4]]
        // B stored as [N×K] = [[5, 7], [6, 8]] (column-major of [[5,6],[7,8]])
        // trans_b=true means read as B^T = [[5,6],[7,8]] (K=2, N=2)
        // A * B^T = [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b_stored = vec![5.0_f64, 7.0, 6.0, 8.0]; // [N=2 × K=2]

        let c = GemmF64::execute_gemm_ex(device, &a, &b_stored, 2, 2, 2, 1, 1.0, 0.0, false, true)
            .unwrap();

        assert_eq!(c.len(), 4);
        assert!(approx_eq(c[0], 19.0, 1e-10));
        assert!(approx_eq(c[1], 22.0, 1e-10));
        assert!(approx_eq(c[2], 43.0, 1e-10));
        assert!(approx_eq(c[3], 50.0, 1e-10));
    }

    #[tokio::test]
    async fn test_gemm_batched() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // Two identity multiplications (3x3)
        let a = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // I_1
            2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, // 2*I_2
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // B_1
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // B_2
        ];

        let c = GemmF64::execute(device, &a, &b, 3, 3, 3, 2).unwrap();

        assert_eq!(c.len(), 18);
        // First batch: I * B = B
        assert!(approx_eq(c[0], 1.0, 1e-10));
        assert!(approx_eq(c[4], 5.0, 1e-10));
        // Second batch: 2I * B = 2B
        assert!(approx_eq(c[9], 2.0, 1e-10));
        assert!(approx_eq(c[13], 10.0, 1e-10));
    }
}
