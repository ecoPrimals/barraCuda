// SPDX-License-Identifier: AGPL-3.0-only
//! Device routing and selection logic.
//!
//! Routes workloads to the appropriate hardware based on the nature of the
//! computation. GPUs run arbitrary WGSL shaders, NPUs run pre-compiled neural
//! network models, CPUs handle everything else.
//!
//! Fallback chain: `preferred → auto-route recommendation → GPU → CPU`

use crate::device::capabilities::{is_gpu_available, is_npu_available};
use crate::device::device_types::Device;

/// Workload hints for automatic device selection.
///
/// Carries enough metadata for the router to make intelligent decisions
/// about data size, sparsity, and hardware affinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadHint {
    /// Large matrix operations (GPU-preferred)
    LargeMatrices,

    /// Small workload (CPU-preferred to avoid GPU dispatch overhead)
    SmallWorkload,

    /// Sparse event processing (NPU-preferred, ultra-low power)
    SparseEvents,

    /// Event-driven logic (CPU or NPU)
    EventProcessing,

    /// String operations (CPU-only)
    StringOps,

    /// General computation (Auto — GPU if available, else CPU)
    General,

    // --- Science-aware hints (route physics/math to the right device) ---
    /// Physics force computation (always GPU — needs WGSL shaders for arbitrary math)
    PhysicsForce,

    /// FFT computation (always GPU — butterfly stages are massively parallel)
    FFT,

    /// Eigenvalue decomposition (GPU for large, CPU for small)
    EigenDecomp,

    /// Linear system solve (GPU for large, CPU for small)
    LinearSolve,

    /// Training / gradient computation (always GPU — needs gradient shaders)
    Training,

    /// Neural network inference with a pre-compiled model (NPU if available)
    Inference,

    /// Binary classification pre-screening (NPU ideal — ultra-low power)
    PreScreen,

    /// Surrogate model evaluation (GPU for RBF kernel, NPU for pre-filter)
    SurrogateEval,

    /// Monte Carlo / random sampling (GPU — parallel PRNG)
    MonteCarlo,

    /// Sparse linear algebra (GPU for large, CPU for small)
    SparseMath,

    /// Reservoir computing / ESN (NPU natural fit — fixed random weights)
    Reservoir,
}

/// Select best device for given workload characteristics (auto-routing).
///
/// Routes workloads to the appropriate hardware. GPUs run arbitrary WGSL
/// shaders, NPUs run pre-compiled neural network models, CPUs handle the rest.
#[must_use]
pub fn select_for_workload(workload: &WorkloadHint) -> Device {
    let gpu = is_gpu_available();
    let npu = is_npu_available();

    match workload {
        // === CPU-only workloads ===
        WorkloadHint::SmallWorkload | WorkloadHint::StringOps => Device::CPU,

        // === NPU-preferred (pre-compiled inference, ultra-low power) ===
        WorkloadHint::SparseEvents if npu => Device::NPU,
        WorkloadHint::EventProcessing if npu => Device::NPU,
        WorkloadHint::PreScreen if npu => Device::NPU,
        WorkloadHint::Inference if npu => Device::NPU,
        WorkloadHint::Reservoir if npu => Device::NPU,

        // === GPU-preferred (arbitrary parallel math) ===
        WorkloadHint::LargeMatrices if gpu => Device::GPU,
        WorkloadHint::PhysicsForce if gpu => Device::GPU,
        WorkloadHint::FFT if gpu => Device::GPU,
        WorkloadHint::EigenDecomp if gpu => Device::GPU,
        WorkloadHint::LinearSolve if gpu => Device::GPU,
        WorkloadHint::Training if gpu => Device::GPU,
        WorkloadHint::SurrogateEval if gpu => Device::GPU,
        WorkloadHint::MonteCarlo if gpu => Device::GPU,
        WorkloadHint::SparseMath if gpu => Device::GPU,

        // === Fallback chain: GPU → CPU ===
        _ => {
            if gpu {
                Device::GPU
            } else {
                Device::CPU
            }
        }
    }
}

/// Select device with an explicit user preference.
///
/// If the user requests a specific device and it is available, honour that
/// choice regardless of what the auto-router would recommend.
///
/// Fallback chain when the preferred device is unavailable:
/// `preferred → auto-route recommendation → GPU → CPU`
#[must_use]
pub fn select_with_preference(preferred: Option<Device>, workload: &WorkloadHint) -> Device {
    match preferred {
        Some(Device::Auto) | None => select_for_workload(workload),
        Some(dev) if device_is_available(dev) => dev,
        Some(_) => select_for_workload(workload),
    }
}

/// Check if a device is available (used by `select_with_preference`).
fn device_is_available(device: Device) -> bool {
    match device {
        Device::CPU => true,
        Device::GPU => is_gpu_available(),
        Device::NPU => is_npu_available(),
        Device::TPU => false,
        Device::Auto => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_workloads_always_route_to_cpu() {
        assert_eq!(
            select_for_workload(&WorkloadHint::SmallWorkload),
            Device::CPU
        );
        assert_eq!(select_for_workload(&WorkloadHint::StringOps), Device::CPU);
    }

    #[test]
    fn general_fallback_never_panics() {
        let device = select_for_workload(&WorkloadHint::General);
        assert!(device == Device::GPU || device == Device::CPU);
    }

    #[test]
    fn preference_none_delegates_to_auto() {
        let auto = select_for_workload(&WorkloadHint::General);
        let pref = select_with_preference(None, &WorkloadHint::General);
        assert_eq!(auto, pref);
    }

    #[test]
    fn preference_auto_delegates_to_auto() {
        let auto = select_for_workload(&WorkloadHint::General);
        let pref = select_with_preference(Some(Device::Auto), &WorkloadHint::General);
        assert_eq!(auto, pref);
    }

    #[test]
    fn preference_cpu_always_honoured() {
        let dev = select_with_preference(Some(Device::CPU), &WorkloadHint::LargeMatrices);
        assert_eq!(dev, Device::CPU);
    }

    #[test]
    fn tpu_unavailable_falls_back() {
        let dev = select_with_preference(Some(Device::TPU), &WorkloadHint::General);
        assert_ne!(dev, Device::TPU);
    }

    #[test]
    fn device_is_available_cpu_always_true() {
        assert!(device_is_available(Device::CPU));
    }

    #[test]
    fn device_is_available_tpu_always_false() {
        assert!(!device_is_available(Device::TPU));
    }

    #[test]
    fn device_is_available_auto_always_true() {
        assert!(device_is_available(Device::Auto));
    }

    #[test]
    fn all_workload_hints_produce_valid_device() {
        let hints = [
            WorkloadHint::LargeMatrices,
            WorkloadHint::SmallWorkload,
            WorkloadHint::SparseEvents,
            WorkloadHint::EventProcessing,
            WorkloadHint::StringOps,
            WorkloadHint::General,
            WorkloadHint::PhysicsForce,
            WorkloadHint::FFT,
            WorkloadHint::EigenDecomp,
            WorkloadHint::LinearSolve,
            WorkloadHint::Training,
            WorkloadHint::Inference,
            WorkloadHint::PreScreen,
            WorkloadHint::SurrogateEval,
            WorkloadHint::MonteCarlo,
            WorkloadHint::SparseMath,
            WorkloadHint::Reservoir,
        ];
        for hint in &hints {
            let dev = select_for_workload(hint);
            assert!(
                matches!(dev, Device::CPU | Device::GPU | Device::NPU),
                "Hint {hint:?} produced unexpected device {dev:?}"
            );
        }
    }
}
