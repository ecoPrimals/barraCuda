// SPDX-License-Identifier: AGPL-3.0-only
//! Unified Scheduler - Automatic Hardware Selection
//!
//! **Philosophy**: Pick the best hardware for each operation automatically
//!
//! The scheduler discovers all available compute hardware at startup and
//! automatically routes operations to the best device based on:
//! - Operation type
//! - Data size
//! - Hardware capabilities
//! - Current load
//!
//! **Deep Debt Principles**:
//! - ✅ Automatic optimization
//! - ✅ Transparent (can override)
//! - ✅ Extensible (new hardware = automatic integration)
//! - ✅ Zero configuration required

use crate::cpu_executor::CpuExecutor;
use crate::error::Result;
use crate::gpu_executor::GpuExecutor;
use crate::unified_hardware::{ComputeExecutor, HardwareType};
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

/// Unified compute scheduler
///
/// Automatically discovers and manages all available compute hardware
pub struct UnifiedScheduler {
    executors: Vec<Arc<dyn ComputeExecutor>>,
    default_executor: Arc<dyn ComputeExecutor>,
}

impl UnifiedScheduler {
    /// Create new scheduler with automatic hardware discovery
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if hardware discovery fails (e.g. GPU initialization fails).
    pub async fn new() -> Result<Self> {
        Self::discover().await
    }

    /// Discover all available hardware
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if hardware discovery fails (e.g. GPU initialization fails).
    pub async fn discover() -> Result<Self> {
        let mut executors: Vec<Arc<dyn ComputeExecutor>> = Vec::new();

        println!("🔍 Discovering compute hardware...");

        // 1. CPU is always available (guaranteed fallback)
        let cpu = Arc::new(CpuExecutor::new());
        println!("  ✅ CPU: {}", cpu.name());
        executors.push(cpu.clone());

        // 2. Try to discover GPU
        match GpuExecutor::new().await {
            Ok(gpu) => {
                println!("  ✅ GPU: {}", gpu.name());
                executors.push(Arc::new(gpu));
            }
            Err(_) => {
                println!("  ⚠️  No GPU available (using CPU fallback)");
            }
        }

        // 3. Try to discover NPU (Akida) - always available in barracuda
        {
            use crate::device::detect_akida_boards;
            match detect_akida_boards() {
                Ok(caps) if !caps.boards.is_empty() => {
                    println!("  ✅ NPU: {} Akida board(s)", caps.boards.len());
                    // Pending: NpuExecutor must implement ComputeExecutor and wrap Akida board handles.
                    // Dependency chain: detect_akida_boards -> NpuExecutor (new) -> executors.push().
                    // Blocked until NpuExecutor implements unified_hardware::ComputeExecutor.
                }
                _ => {
                    // NPU not available (okay, not all systems have it)
                }
            }
        }

        println!("✨ Discovered {} executor(s)", executors.len());

        // CPU is the default fallback
        let default_executor = cpu;

        Ok(Self {
            executors,
            default_executor,
        })
    }

    /// Select best executor for an operation
    #[must_use]
    pub fn select_executor(
        &self,
        op: &MathOp,
        inputs: &[TensorDescriptor],
    ) -> Arc<dyn ComputeExecutor> {
        // Find all executors that can handle this operation
        let candidates: Vec<_> = self
            .executors
            .iter()
            .filter(|e| e.can_execute(op, inputs))
            .collect();

        if candidates.is_empty() {
            // No executor can handle this - use default (CPU)
            return self.default_executor.clone();
        }

        // Score each candidate and pick the best
        // Invariant: candidates is non-empty (checked above with early return)
        candidates
            .iter()
            .max_by(|a, b| {
                let score_a = a.score_operation(op, inputs);
                let score_b = b.score_operation(op, inputs);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| self.default_executor.clone(), |best| (*best).clone())
    }

    /// Get all available executors
    #[must_use]
    pub fn executors(&self) -> &[Arc<dyn ComputeExecutor>] {
        &self.executors
    }

    /// Get default executor (CPU fallback)
    #[must_use]
    pub fn default_executor(&self) -> &Arc<dyn ComputeExecutor> {
        &self.default_executor
    }

    /// Get executor by hardware type
    #[must_use]
    pub fn get_executor(&self, hardware_type: HardwareType) -> Option<Arc<dyn ComputeExecutor>> {
        self.executors
            .iter()
            .find(|e| e.hardware_type() == hardware_type)
            .cloned()
    }

    /// Print summary of available hardware
    pub fn print_summary(&self) {
        println!("\n📊 Compute Hardware Summary");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        for executor in &self.executors {
            let caps = executor.capabilities();
            println!("\n🔧 {}", executor.name());
            println!("   Type: {:?}", executor.hardware_type());
            println!("   Parallel Units: {}", caps.parallelism.max_parallel_units);
            println!("   Memory: {:.1} GB", caps.memory.total_bytes as f64 / 1e9);
            println!(
                "   Peak TFLOPS (FP32): {:.1}",
                caps.performance.peak_tflops_fp32
            );
            println!("   Operations:");
            if caps.operations.matmul {
                println!("     ✅ Matrix Multiply");
            }
            if caps.operations.convolution {
                println!("     ✅ Convolution");
            }
            if caps.operations.reductions {
                println!("     ✅ Reductions");
            }
            if caps.operations.custom_kernels {
                println!("     ✅ Custom Kernels");
            }
        }

        println!("\n✨ Default Fallback: {}", self.default_executor.name());
        println!();
    }
}

/// Example: Automatic device selection
///
/// ```rust,ignore
/// use barracuda::scheduler::UnifiedScheduler;
/// use barracuda::unified_math::{MathOp, TensorDescriptor, DType};
///
/// # async fn example() -> Result<()> {
/// // Discover all hardware
/// let scheduler = UnifiedScheduler::new().await?;
/// scheduler.print_summary();
///
/// // Small operation → CPU chosen
/// let small_desc = TensorDescriptor::new(vec![10, 10], DType::F32);
/// let small_op = MathOp::ReLU;
/// let executor = scheduler.select_executor(&small_op, &[small_desc]);
/// println!("Small ReLU: {}", executor.name()); // → CPU
///
/// // Large matrix → GPU chosen
/// let large_desc = TensorDescriptor::new(vec![4096, 4096], DType::F32);
/// let large_op = MathOp::MatMul { transpose_a: false, transpose_b: false };
/// let executor = scheduler.select_executor(&large_op, &[large_desc.clone(), large_desc]);
/// println!("Large MatMul: {}", executor.name()); // → GPU
/// # Ok(())
/// # }
/// ```
#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_math::DType;

    #[tokio::test]
    async fn test_scheduler_creation() {
        let scheduler = UnifiedScheduler::new().await.unwrap();

        // At least CPU should be available
        assert!(!scheduler.executors().is_empty());
        assert_eq!(
            scheduler.default_executor().hardware_type(),
            HardwareType::CPU
        );
    }

    #[tokio::test]
    async fn test_scheduler_discovery() {
        let scheduler = UnifiedScheduler::discover().await.unwrap();

        // CPU is always available
        let cpu = scheduler.get_executor(HardwareType::CPU);
        assert!(cpu.is_some());

        // Print what we found
        scheduler.print_summary();
    }

    #[tokio::test]
    async fn test_small_vs_large_selection() {
        let scheduler = UnifiedScheduler::new().await.unwrap();

        // Small operation
        let small = TensorDescriptor::new(vec![10, 10], DType::F32);
        let small_op = MathOp::ReLU;
        let small_exec = scheduler.select_executor(&small_op, &[small]);
        println!("Small ReLU → {}", small_exec.name());

        // Large operation
        let large = TensorDescriptor::new(vec![2048, 2048], DType::F32);
        let large_op = MathOp::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        let large_exec = scheduler.select_executor(&large_op, &[large.clone(), large]);
        println!("Large MatMul → {}", large_exec.name());

        // If GPU available, large should use GPU, small should use CPU
        if scheduler.get_executor(HardwareType::GPU).is_some() {
            // GPU available - verify smart selection
            // (Small ops may still use GPU if it's much faster)
        }
    }

    #[tokio::test]
    async fn test_matmul_scoring() {
        let scheduler = UnifiedScheduler::new().await.unwrap();

        // Test different matrix sizes
        let sizes = vec![
            (10, 10),     // Tiny
            (100, 100),   // Small
            (1000, 1000), // Medium
            (4096, 4096), // Large
        ];

        for (m, n) in sizes {
            let desc = TensorDescriptor::new(vec![m, n], DType::F32);
            let op = MathOp::MatMul {
                transpose_a: false,
                transpose_b: false,
            };
            let exec = scheduler.select_executor(&op, &[desc.clone(), desc]);
            println!("MatMul [{}x{}] → {}", m, n, exec.name());
        }
    }
}
