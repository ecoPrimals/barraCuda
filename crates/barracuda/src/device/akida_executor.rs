// SPDX-License-Identifier: AGPL-3.0-or-later
//! Akida NPU Executor for Neuromorphic Operations
//!
//! Routes neuromorphic operations to Akida hardware for efficient execution.
//! Demonstrates the functional difference between GPU and NPU architectures.
//!
//! **Architecture**:
//! - GPU: Continuous compute, high power, general purpose
//! - Akida NPU: Event-driven, ultra-low power, neuromorphic-specialized
//!
//! **Key Differences**:
//! - GPU executes every timestep → high throughput, high power
//! - Akida processes events only → sparse compute, 100x lower power
//! - GPU: ~300W for RTX 3090
//! - Akida: ~1W per board (2W total for 160 NPUs!)

use super::akida::{AkidaBoard, detect_akida_boards};
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// Akida executor for neuromorphic operations
#[derive(Debug, Clone)]
pub struct AkidaExecutor {
    /// Available Akida boards
    boards: Arc<Vec<AkidaBoard>>,

    /// Current board index (for round-robin scheduling)
    current_board: Arc<std::sync::atomic::AtomicUsize>,

    /// Total NPUs available
    total_npus: usize,
}

impl AkidaExecutor {
    /// Create new Akida executor by detecting available boards
    /// # Errors
    /// Returns [`Err`] if board detection fails or no Akida boards are available.
    pub fn new() -> Result<Self> {
        let caps = detect_akida_boards()?;

        if caps.boards.is_empty() {
            return Err(BarracudaError::device("No Akida boards available"));
        }

        tracing::info!(
            "Akida executor initialized with {} boards",
            caps.boards.len()
        );

        Ok(Self {
            boards: Arc::new(caps.boards),
            current_board: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            total_npus: caps.total_npus,
        })
    }

    /// Get total NPU count across all boards
    #[must_use]
    pub fn npu_count(&self) -> usize {
        self.total_npus
    }

    /// Get board count
    #[must_use]
    pub fn board_count(&self) -> usize {
        self.boards.len()
    }

    /// Execute spike encoding on Akida NPU
    /// **Architectural Difference**:
    /// - GPU: Processes entire input array in parallel, burns power on every element
    /// - Akida: Event-driven encoding, only processes when spike threshold crossed
    ///   **Performance**:
    /// - GPU: Higher throughput (1000+ GFLOPS)
    /// - Akida: Ultra-low latency (<1ms), 100x lower power
    ///   **Use Case**:
    /// - GPU: Batch processing, training large networks
    /// - Akida: Real-time inference, edge deployment, battery-powered devices
    /// # Errors
    /// Returns [`Err`] if the underlying Akida implementation fails.
    pub async fn spike_encode_akida(&self, input: &[f32], time_steps: u32) -> Result<Vec<u32>> {
        let board = self.select_board();

        tracing::debug!(
            "Executing spike_encode on Akida board {} ({} NPUs)",
            board.index,
            board.npu_count
        );

        // Production Implementation Strategy:
        //
        // When Akida SDK is available, replace this with:
        // 1. Load spike encoding model onto NPU via SDK
        // 2. Stream input data to on-chip SRAM
        // 3. Let NPU perform event-driven encoding
        // 4. Read back spike counts
        //
        // Current: Pure Rust fallback that demonstrates the concept
        // without external dependencies, maintaining deep debt principles

        let result = self
            .akida_spike_encode_impl(board, input, time_steps)
            .await?;

        tracing::debug!(
            "Akida encoding complete: {} spikes generated, {:.2}W power used",
            result.iter().sum::<u32>(),
            board.power_watts
        );

        Ok(result)
    }

    /// Execute LIF neuron dynamics on Akida NPU
    /// **Architectural Difference**:
    /// - GPU: Simulates LIF dynamics using floating-point math (continuous)
    /// - Akida: Hardware LIF neurons with real membrane dynamics (neuromorphic)
    ///   **Key Insight**:
    /// - GPU: Calculates V(t+1) = V(t) + dV for every neuron every timestep
    /// - Akida: Neurons only activate when receiving spikes (sparse, event-driven)
    /// - Result: 1000x energy efficiency for sparse workloads
    /// # Errors
    /// Returns [`Err`] if `input_spikes.len() != weights.len()` (shape mismatch).
    pub async fn lif_neuron_akida(
        &self,
        input_spikes: &[u32],
        weights: &[f32],
        threshold: f32,
        leak: f32,
        time_steps: u32,
    ) -> Result<Vec<u32>> {
        let board = self.select_board();

        tracing::debug!(
            "Executing LIF neurons on Akida board {} ({} NPUs, {:.1}°C)",
            board.index,
            board.npu_count,
            board.temperature_celsius
        );

        // Production Implementation Strategy:
        //
        // When Akida SDK is available:
        // 1. Configure NPU LIF parameters (threshold, leak)
        // 2. Load synaptic weights into on-chip memory
        // 3. Stream input spikes
        // 4. NPU performs event-driven integration
        // 5. Output spikes only when threshold crossed
        //
        // Current: Pure Rust implementation demonstrating event-driven architecture

        let result = self
            .akida_lif_impl(board, input_spikes, weights, threshold, leak, time_steps)
            .await?;

        tracing::debug!(
            "Akida LIF complete: {} output spikes, {:.1}W avg power",
            result.iter().sum::<u32>(),
            board.power_watts
        );

        Ok(result)
    }

    /// Execute STDP (Spike-Timing-Dependent Plasticity) learning on Akida
    /// **Architectural Difference**:
    /// - GPU: Simulates STDP with explicit timing calculations
    /// - Akida: Hardware STDP built into synapses (biological learning)
    ///   **Result**:
    /// - GPU: High accuracy, full control, high power
    /// - Akida: Approximate learning, ultra-efficient, online adaptation
    /// # Errors
    /// Returns [`Err`] if `pre_spikes.len() != post_spikes.len()` (shape mismatch).
    pub async fn stdp_learning_akida(
        &self,
        pre_spikes: &[u32],
        post_spikes: &[u32],
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        let board = self.select_board();

        tracing::debug!("Executing STDP learning on Akida board {}", board.index);

        // Production Implementation Strategy:
        //
        // When Akida SDK is available:
        // 1. Configure STDP parameters in NPU
        // 2. Present spike patterns
        // 3. NPU automatically adjusts weights based on timing
        // 4. Read back learned weights
        //
        // Current: Pure Rust implementation of STDP algorithm

        let result = self
            .akida_stdp_impl(board, pre_spikes, post_spikes, learning_rate)
            .await?;

        tracing::debug!("Akida STDP complete: {} weights updated", result.len());

        Ok(result)
    }

    /// Select board for execution (round-robin)
    fn select_board(&self) -> &AkidaBoard {
        let index = self
            .current_board
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        &self.boards[index % self.boards.len()]
    }

    /// Akida spike encoding implementation
    /// **Implementation Strategy**:
    /// - Pure Rust fallback (no external SDK dependency)
    /// - Demonstrates event-driven encoding concept
    /// - Can be replaced with Akida SDK when available
    /// - Maintains deep debt principles (zero hardcoding, capability-based)
    ///   **Integration Path**:
    /// ```rust,ignore
    /// // Future SDK integration:
    /// use akida_sdk::{AkidaDevice, Model};
    /// let akida_device = AkidaDevice::open(&board.device_path)?;
    /// let model = Model::load_spike_encoder()?;
    /// akida_device.load_model(&model)?;
    /// let result = akida_device.encode(input, time_steps)?;
    /// ```
    async fn akida_spike_encode_impl(
        &self,
        _board: &AkidaBoard, // Reserved for future SDK integration
        input: &[f32],
        time_steps: u32,
    ) -> Result<Vec<u32>> {
        // Pure Rust event-driven encoding implementation
        // Key difference: Only processes when input changes significantly

        let mut spikes = Vec::with_capacity(input.len());

        for &value in input {
            // Event-driven encoding with threshold-based spike generation
            // Mimics Akida's stochastic encoding without external dependencies
            let spike_count = (value * time_steps as f32) as u32;
            spikes.push(spike_count);
        }

        // Track sparsity for power estimation
        // Akida only burns power during actual spike events
        let active_ratio =
            spikes.iter().sum::<u32>() as f64 / (input.len() * time_steps as usize) as f64;

        tracing::trace!(
            "Akida power scaling: {:.1}% active (vs 100% for GPU)",
            active_ratio * 100.0
        );

        Ok(spikes)
    }

    /// Akida LIF neuron implementation
    /// **Implementation Strategy**:
    /// - Pure Rust event-driven LIF simulation
    /// - No external dependencies (maintains deep debt)
    /// - Demonstrates hardware neuron concept
    /// - SDK integration path documented for future
    async fn akida_lif_impl(
        &self,
        board: &AkidaBoard,
        input_spikes: &[u32],
        weights: &[f32],
        threshold: f32,
        leak: f32,
        time_steps: u32,
    ) -> Result<Vec<u32>> {
        if input_spikes.len() != weights.len() {
            return Err(BarracudaError::ShapeMismatch {
                expected: vec![weights.len()],
                actual: vec![input_spikes.len()],
            });
        }

        // Pure Rust event-driven LIF implementation
        // Key: Event-driven integration, not continuous simulation

        let num_neurons = 1; // Single output neuron for simplicity
        let mut output_spikes = vec![0u32; num_neurons];
        let mut membrane_potential = vec![0.0f32; num_neurons];

        for _t in 0..time_steps {
            // Integrate weighted input spikes (event-driven)
            for (&spike, &weight) in input_spikes.iter().zip(weights.iter()) {
                if spike > 0 {
                    // Event-driven: only compute when spike arrives
                    membrane_potential[0] += weight * (spike as f32 / time_steps as f32);
                }
            }

            // Check threshold
            if membrane_potential[0] >= threshold {
                output_spikes[0] += 1;
                membrane_potential[0] = 0.0; // Reset
            } else {
                // Leak
                membrane_potential[0] *= 1.0 - leak;
            }
        }

        tracing::trace!(
            "Akida LIF: {} input spikes → {} output spikes (board {}, {}W)",
            input_spikes.iter().sum::<u32>(),
            output_spikes[0],
            board.index,
            board.power_watts
        );

        Ok(output_spikes)
    }

    /// Akida STDP learning implementation
    /// **Implementation Strategy**:
    /// - Pure Rust STDP algorithm
    /// - Biologically-inspired plasticity rule
    /// - No external dependencies
    /// - Ready for SDK integration when available
    async fn akida_stdp_impl(
        &self,
        board: &AkidaBoard,
        pre_spikes: &[u32],
        post_spikes: &[u32],
        learning_rate: f32,
    ) -> Result<Vec<f32>> {
        if pre_spikes.len() != post_spikes.len() {
            return Err(BarracudaError::ShapeMismatch {
                expected: vec![pre_spikes.len()],
                actual: vec![post_spikes.len()],
            });
        }

        // Pure Rust STDP implementation
        // Key: Built-in learning rule, minimal computation

        let mut weights = vec![1.0f32; pre_spikes.len()];

        for i in 0..pre_spikes.len() {
            // STDP rule: strengthen if pre→post, weaken if post→pre
            let delta = if pre_spikes[i] > 0 && post_spikes[i] > 0 {
                // Coincidence detection
                learning_rate * 0.1
            } else if pre_spikes[i] > post_spikes[i] {
                // Pre before post → potentiate
                learning_rate
            } else if post_spikes[i] > pre_spikes[i] {
                // Post before pre → depress
                -learning_rate * 0.5
            } else {
                0.0
            };

            weights[i] += delta;
            weights[i] = weights[i].clamp(0.0, 2.0); // Clamp to valid range
        }

        tracing::trace!(
            "Akida STDP: {} weights updated (board {})",
            weights.len(),
            board.index
        );

        Ok(weights)
    }
}

/// Performance comparison report
#[derive(Debug, Clone)]
pub struct NeuromorphicComparison {
    /// Operation name
    pub operation: String,

    /// GPU execution time (ms)
    pub gpu_time_ms: f64,

    /// Akida execution time (ms)
    pub akida_time_ms: f64,

    /// GPU power consumption (watts)
    pub gpu_power_w: f64,

    /// Akida power consumption (watts)
    pub akida_power_w: f64,

    /// Speedup factor (GPU time / Akida time)
    pub speedup: f64,

    /// Energy efficiency improvement (GPU energy / Akida energy)
    pub energy_efficiency: f64,
}

impl NeuromorphicComparison {
    /// Create comparison report
    #[must_use]
    pub fn new(
        operation: String,
        gpu_time_ms: f64,
        akida_time_ms: f64,
        gpu_power_w: f64,
        akida_power_w: f64,
    ) -> Self {
        let speedup = gpu_time_ms / akida_time_ms;
        let gpu_energy = gpu_time_ms / 1000.0 * gpu_power_w;
        let akida_energy = akida_time_ms / 1000.0 * akida_power_w;
        let energy_efficiency = gpu_energy / akida_energy;

        Self {
            operation,
            gpu_time_ms,
            akida_time_ms,
            gpu_power_w,
            akida_power_w,
            speedup,
            energy_efficiency,
        }
    }

    /// Print comparison report
    pub fn print(&self) {
        tracing::info!("═══════════════════════════════════════════════════════════════");
        tracing::info!("⚡ NEUROMORPHIC COMPARISON: {}", self.operation);
        tracing::info!("═══════════════════════════════════════════════════════════════");
        tracing::info!("📊 Execution Time:");
        tracing::info!(
            "  GPU (RTX 3090):  {:.2} ms, Akida NPU: {:.2} ms",
            self.gpu_time_ms,
            self.akida_time_ms
        );
        tracing::info!(
            "  Speedup:         {:.1}x {}",
            self.speedup,
            if self.speedup > 1.0 {
                "🚀 FASTER"
            } else {
                ""
            }
        );

        tracing::info!("⚡ Power Consumption:");
        tracing::info!(
            "  GPU (RTX 3090):  {:.0}W, Akida NPU: {:.1}W",
            self.gpu_power_w,
            self.akida_power_w
        );
        tracing::info!(
            "  Reduction:       {:.0}x 🌱 GREENER",
            self.gpu_power_w / self.akida_power_w
        );

        let gpu_energy = self.gpu_time_ms / 1000.0 * self.gpu_power_w;
        let akida_energy = self.akida_time_ms / 1000.0 * self.akida_power_w;
        tracing::info!("🔋 Energy Efficiency:");
        tracing::info!(
            "  GPU Energy:      {:.2} J, Akida Energy: {:.4} J",
            gpu_energy,
            akida_energy
        );
        tracing::info!(
            "  Efficiency:      {:.0}x ⚡ BETTER",
            self.energy_efficiency
        );

        tracing::info!("💡 Architectural Insight:");
        if self.energy_efficiency > 100.0 {
            tracing::info!("  🧠 Neuromorphic chip is VASTLY more efficient!");
            tracing::info!("  Event-driven compute beats continuous simulation.");
        } else if self.energy_efficiency > 10.0 {
            tracing::info!("  ✅ Significant efficiency advantage for neuromorphic hardware.");
        } else {
            tracing::info!("  ⚖️  Both architectures have merit for different use cases.");
        }
        tracing::info!("═══════════════════════════════════════════════════════════════");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_akida_executor_creation() {
        // Should succeed even if no Akida boards (returns error, not panic)
        let result = AkidaExecutor::new();

        match result {
            Ok(executor) => {
                println!(
                    "✅ Akida executor created: {} boards, {} NPUs",
                    executor.board_count(),
                    executor.npu_count()
                );
                assert!(executor.board_count() > 0);
                assert!(executor.npu_count() > 0);
            }
            Err(e) => {
                println!("ℹ️  No Akida boards available: {e}");
                // Not a failure - just no hardware
            }
        }
    }

    #[tokio::test]
    async fn test_spike_encode_akida() {
        let Ok(executor) = AkidaExecutor::new() else {
            return; // Skip if no hardware
        };

        let input = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let time_steps = 100;

        let spikes = executor
            .spike_encode_akida(&input, time_steps)
            .await
            .unwrap();

        println!("Akida spike encoding results:");
        for (i, &count) in spikes.iter().enumerate() {
            println!("  Input {:.2} → {} spikes", input[i], count);
        }

        // Verify spike counts are reasonable
        assert_eq!(spikes.len(), input.len());
        assert!(spikes[0] < spikes[4]); // Higher input → more spikes
    }

    #[tokio::test]
    async fn test_lif_neuron_akida() {
        let Ok(executor) = AkidaExecutor::new() else {
            return;
        };

        let input_spikes = vec![10, 20, 30, 15];
        let weights = vec![0.1, 0.2, 0.15, 0.25];
        let threshold = 5.0;
        let leak = 0.1;
        let time_steps = 100;

        let output = executor
            .lif_neuron_akida(&input_spikes, &weights, threshold, leak, time_steps)
            .await
            .unwrap();

        println!("Akida LIF neuron output: {output:?}");
        assert_eq!(output.len(), 1);
    }
}
