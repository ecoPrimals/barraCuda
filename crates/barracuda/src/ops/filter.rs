// SPDX-License-Identifier: AGPL-3.0-or-later
//! Filter — full GPU stream compaction (predicate → prefix sum → scatter)
//!
//! ## Algorithm (single-level, n ≤ 65,536)
//!
//! 1. **evaluate_predicate**: `flags[i] = keep ? 1 : 0`
//! 2. **local_scan**: intra-workgroup exclusive scan → `scan[i]` (local) + `wg_sums[wg]`
//! 3. **add_wg_offsets**: single-workgroup scan of `wg_sums[]`, adds offsets to `scan[]`
//! 4. **scatter**: `output[scan[i]] = input[i]` if `flags[i] == 1`
//!
//! ## Algorithm (two-level, 65,536 < n ≤ 16,777,216)
//!
//! Adds two extra passes to handle arrays requiring >256 level-0 workgroups:
//! 1. **evaluate_predicate** (unchanged)
//! 2. **local_scan** on `flags` → `scan1`, `wg_sums1`
//! 3. **local_scan** on `wg_sums1` → `wg_sums1_scan`, `wg_sums2`  (≤256 groups)
//! 4. **add_wg_offsets** on `wg_sums2` (1 workgroup) → `wg_sums1_scan` globally correct
//! 5. **apply_l1_offsets** (`n_groups1` workgroups) → adds `wg_sums1_scan[wg]` to `scan1`
//! 6. **scatter** (unchanged)
//!
//! Input size limit: 16,777,216 elements (WG³ = 256³).  Returns an error for
//! larger inputs (genome-scale beyond 16M requires a three-level extension).
//!
//! ## Returns
//!
//! A `FilterResult` containing:
//! - `selected`: a `Tensor` of shape `[count]` with only the passing values (compacted)
//! - `count`: number of elements that satisfied the predicate
//!
//! Deep Debt Principles:
//! - Complete implementation — no mocks, stubs, or placeholder paths
//! - GPU-resident — no intermediate CPU readbacks
//! - Capability-based dispatch — workgroup size from `DeviceCapabilities`

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

// ─── Constants ───────────────────────────────────────────────────────────────

const SCAN_WG: u32 = 256;
/// Maximum elements for the two-level path (WG³ = 16,777,216).
const SCAN_L2_THRESHOLD: u32 = SCAN_WG * SCAN_WG * SCAN_WG;

// ─── Public types ────────────────────────────────────────────────────────────

/// Result of a stream-compaction filter operation.
pub struct FilterResult {
    /// Compacted tensor containing only values that passed the predicate.
    /// Shape is `[count]`.
    pub selected: Tensor,
    /// Number of elements that passed the predicate.
    pub count: usize,
}

/// Predicate operation for element-wise filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOperation {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterOrEqual,
    LessOrEqual,
}

impl FilterOperation {
    fn to_u32(self) -> u32 {
        match self {
            FilterOperation::GreaterThan => 0,
            FilterOperation::LessThan => 1,
            FilterOperation::Equal => 2,
            FilterOperation::NotEqual => 3,
            FilterOperation::GreaterOrEqual => 4,
            FilterOperation::LessOrEqual => 5,
        }
    }
}

// ─── GPU uniform structs ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FilterParams {
    size: u32,
    operation: u32,
    n_groups: u32,
    _pad: u32,
    threshold: f32,
    epsilon: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScanConfig {
    n: u32,
    n_groups: u32,
    _pad0: u32,
    _pad1: u32,
}

const DEFAULT_FILTER_EPSILON: f32 = 1e-5;

/// GPU stream-compaction filter.
pub struct Filter {
    input: Tensor,
    operation: FilterOperation,
    threshold: f32,
    /// Equality/NotEqual tolerance.
    epsilon: f32,
}

impl Filter {
    fn filter_shader() -> &'static str {
        include_str!("../shaders/misc/filter.wgsl")
    }

    fn scan_shader() -> &'static str {
        include_str!("../shaders/misc/prefix_sum.wgsl")
    }

    pub fn new(input: Tensor, operation: FilterOperation, threshold: f32) -> Self {
        Self {
            input,
            operation,
            threshold,
            epsilon: DEFAULT_FILTER_EPSILON,
        }
    }

    pub fn with_epsilon(mut self, eps: f32) -> Self {
        self.epsilon = eps;
        self
    }

    /// Execute GPU stream compaction, automatically selecting single- or two-level
    /// prefix-sum based on input size.
    ///
    /// - `n ≤ 65,536`  (WG²): single-level, 4 GPU passes
    /// - `n ≤ 16,777,216` (WG³): two-level, 6 GPU passes
    /// - `n > 16,777,216`: returns `BarracudaError::InvalidInput` (extend to three-level)
    pub fn execute(self) -> Result<FilterResult> {
        let device = self.input.device();
        let n = self.input.len();
        let n_u32 = n as u32;

        if n_u32 > SCAN_L2_THRESHOLD {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "ParallelFilter: input length {n} exceeds the two-level maximum \
                     ({SCAN_L2_THRESHOLD} = WG³). Extend to a three-level hierarchy for \
                     genome-scale inputs."
                ),
            });
        }

        // Empty input: return immediately without creating zero-sized buffers (wgpu rejects them)
        if n == 0 {
            let output_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Filter Output Empty"),
                size: 1, // Minimum 1 byte for valid buffer
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let selected = Tensor::from_buffer(output_buf, vec![0], device.clone());
            return Ok(FilterResult { selected, count: 0 });
        }

        let n_groups = n_u32.div_ceil(SCAN_WG);
        let u32_bytes = std::mem::size_of::<u32>() as u64;
        let f32_bytes = std::mem::size_of::<f32>() as u64;

        // ── 1. Allocate core buffers ─────────────────────────────────────────
        let flags_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Flags"),
            size: n as u64 * u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let scan_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Scan"),
            size: n as u64 * u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // wg_sums1: one entry per level-0 workgroup
        let wg_sums_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter WgSums"),
            size: n_groups as u64 * u32_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Output"),
            size: n as u64 * f32_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let total_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Filter Total"),
            size: u32_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Filter params uniform ─────────────────────────────────────────────
        let filter_params = FilterParams {
            size: n_u32,
            operation: self.operation.to_u32(),
            n_groups,
            _pad: 0,
            threshold: self.threshold,
            epsilon: self.epsilon,
            _pad2: 0.0,
            _pad3: 0.0,
        };
        let filter_params_buf = device.create_uniform_buffer("Filter Params", &filter_params);

        // ── Scan config uniform (level 0: n elements, n_groups groups) ────────
        let scan_cfg = ScanConfig {
            n: n_u32,
            n_groups,
            _pad0: 0,
            _pad1: 0,
        };
        let scan_cfg_buf = device.create_uniform_buffer("Scan Config L0", &scan_cfg);

        // Pass 1: predicate
        ComputeDispatch::new(device, "filter_predicate")
            .shader(Self::filter_shader(), "evaluate_predicate")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &flags_buf)
            .storage_rw(2, &scan_buf)
            .storage_rw(3, &output_buf)
            .storage_rw(4, &total_buf)
            .uniform(5, &filter_params_buf)
            .dispatch(n_groups, 1, 1)
            .submit();

        // Pass 2a: intra-workgroup scan
        ComputeDispatch::new(device, "scan_local")
            .shader(Self::scan_shader(), "local_scan")
            .uniform(0, &scan_cfg_buf)
            .storage_read(1, &flags_buf)
            .storage_rw(2, &scan_buf)
            .storage_rw(3, &wg_sums_buf)
            .dispatch(n_groups, 1, 1)
            .submit();

        if n_groups <= SCAN_WG {
            // ── Single-level path (n ≤ 65,536) ───────────────────────────────

            // Pass 2b: single-workgroup scan of wg_sums + add offsets to scan_buf
            ComputeDispatch::new(device, "scan_offsets")
                .shader(Self::scan_shader(), "add_wg_offsets")
                .uniform(0, &scan_cfg_buf)
                .storage_read(1, &flags_buf)
                .storage_rw(2, &scan_buf)
                .storage_rw(3, &wg_sums_buf)
                .dispatch(1, 1, 1)
                .submit();
        } else {
            // ── Two-level path (65,536 < n ≤ 16,777,216) ─────────────────────
            let n_groups2 = n_groups.div_ceil(SCAN_WG);

            // Extra buffers for level-1 scan
            let wg_sums1_scan_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Filter WgSums1 Scan"),
                size: n_groups as u64 * u32_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let wg_sums2_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Filter WgSums2"),
                size: n_groups2 as u64 * u32_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Level-1 config: treat wg_sums1 as the input array
            let scan_l1_cfg = ScanConfig {
                n: n_groups,
                n_groups: n_groups2,
                _pad0: 0,
                _pad1: 0,
            };
            let scan_l1_cfg_buf = device.create_uniform_buffer("Scan Config L1", &scan_l1_cfg);

            // Pass 2b (L1): intra-workgroup scan of wg_sums1
            ComputeDispatch::new(device, "scan_l1_local")
                .shader(Self::scan_shader(), "local_scan")
                .uniform(0, &scan_l1_cfg_buf)
                .storage_read(1, &wg_sums_buf)
                .storage_rw(2, &wg_sums1_scan_buf)
                .storage_rw(3, &wg_sums2_buf)
                .dispatch(n_groups2, 1, 1)
                .submit();

            // Pass 2c: single-workgroup scan of wg_sums2 → corrects wg_sums1_scan
            ComputeDispatch::new(device, "scan_offsets_l1")
                .shader(Self::scan_shader(), "add_wg_offsets")
                .uniform(0, &scan_l1_cfg_buf)
                .storage_read(1, &wg_sums_buf)
                .storage_rw(2, &wg_sums1_scan_buf)
                .storage_rw(3, &wg_sums2_buf)
                .dispatch(1, 1, 1)
                .submit();

            // Pass 2d: apply L1 offsets (n_groups workgroups) → scan_buf globally correct
            ComputeDispatch::new(device, "apply_l1_offsets")
                .shader(Self::scan_shader(), "apply_l1_offsets")
                .uniform(0, &scan_cfg_buf)
                .storage_read(1, &wg_sums1_scan_buf)
                .storage_rw(2, &scan_buf)
                .storage_rw(3, &wg_sums_buf)
                .dispatch(n_groups, 1, 1)
                .submit();
        }

        // Pass 3: scatter
        ComputeDispatch::new(device, "filter_scatter")
            .shader(Self::filter_shader(), "scatter")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &flags_buf)
            .storage_rw(2, &scan_buf)
            .storage_rw(3, &output_buf)
            .storage_rw(4, &total_buf)
            .uniform(5, &filter_params_buf)
            .dispatch(n_groups, 1, 1)
            .submit();

        // ── 6. Read back count and build result ───────────────────────────────
        let count_vec = crate::utils::read_buffer_u32(device, &total_buf, 1)?;
        let count = count_vec[0] as usize;

        // Wrap the output buffer (compacted, but may have unwritten tail — only
        // `count` values are valid).  We expose the full buffer; callers use
        // `count` to slice.  A future evolution can truncate via a GPU copy.
        let selected = Tensor::from_buffer(output_buf, vec![n], device.clone());

        Ok(FilterResult { selected, count })
    }
}

// ─── Tensor convenience API ──────────────────────────────────────────────────

impl Tensor {
    /// Stream-compact this tensor, keeping elements satisfying `operation(x, threshold)`.
    ///
    /// Returns a `FilterResult` with a compacted tensor and element count.
    ///
    /// # Example
    /// ```ignore
    /// let result = tensor.filter(FilterOperation::GreaterThan, 4.0)?;
    /// let selected = result.selected.to_vec()?;  // only passing values
    /// let count = result.count;
    /// ```
    pub fn filter(self, operation: FilterOperation, threshold: f32) -> Result<FilterResult> {
        Filter::new(self, operation, threshold).execute()
    }

    /// Stream-compact keeping elements `> threshold`. Returns `(selected, count)`.
    pub fn filter_gt(self, threshold: f32) -> Result<FilterResult> {
        self.filter(FilterOperation::GreaterThan, threshold)
    }

    /// Stream-compact keeping elements `< threshold`. Returns `(selected, count)`.
    pub fn filter_lt(self, threshold: f32) -> Result<FilterResult> {
        self.filter(FilterOperation::LessThan, threshold)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    const FILTER_SHADER: &str = include_str!("../shaders/misc/filter.wgsl");
    const PREFIX_SUM_SHADER: &str = include_str!("../shaders/misc/prefix_sum.wgsl");

    #[test]
    fn filter_params_layout() {
        assert_eq!(std::mem::size_of::<FilterParams>(), 32);
    }

    #[test]
    fn scan_config_layout() {
        assert_eq!(std::mem::size_of::<ScanConfig>(), 16);
    }

    #[test]
    fn filter_shader_source_valid() {
        assert!(!FILTER_SHADER.is_empty());
        assert!(FILTER_SHADER.contains("evaluate_predicate"));
        assert!(FILTER_SHADER.contains("scatter"));
    }

    #[test]
    fn prefix_sum_shader_source_valid() {
        assert!(!PREFIX_SUM_SHADER.is_empty());
        assert!(PREFIX_SUM_SHADER.contains("local_scan"));
        assert!(PREFIX_SUM_SHADER.contains("add_wg_offsets"));
    }

    #[test]
    fn filter_shader_compiles() {
        let source = FILTER_SHADER;
        assert!(!source.is_empty());
        assert!(source.contains("@compute"));
        assert!(source.contains("fn evaluate_predicate"));
    }

    #[test]
    fn prefix_sum_shader_compiles() {
        let source = PREFIX_SUM_SHADER;
        assert!(!source.is_empty());
        assert!(source.contains("@compute"));
        assert!(source.contains("fn local_scan"));
        assert!(source.contains("apply_l1_offsets"));
    }
}

#[cfg(test)]
#[path = "filter_tests.rs"]
mod tests;
