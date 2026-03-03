//! Driver capability detection — f64 workarounds, NVK/RADV detection

use super::WgpuDevice;

impl WgpuDevice {
    /// Check if this device uses the NVK (nouveau) Vulkan driver
    ///
    /// NVK is the open-source Vulkan driver for NVIDIA GPUs built on Mesa.
    /// Some f64 builtins (particularly `exp()`) crash the NAK compiler on NVK.
    pub fn is_nvk(&self) -> bool {
        let driver = self.adapter_info.driver.to_lowercase();
        let driver_info = self.adapter_info.driver_info.to_lowercase();
        driver.contains("nvk")
            || driver.contains("nouveau")
            || driver.contains("mesa")
            || driver_info.contains("nvk")
            || driver_info.contains("nouveau")
    }

    /// Check if this device uses AMD's RADV Vulkan driver
    ///
    /// RADV is the open-source Vulkan driver for AMD GPUs built on Mesa.
    pub fn is_radv(&self) -> bool {
        let driver = self.adapter_info.driver.to_lowercase();
        let driver_info = self.adapter_info.driver_info.to_lowercase();
        driver.contains("radv") || driver_info.contains("radv")
    }

    /// Whether this device needs software workarounds for f64 exp/log builtins.
    ///
    /// Known broken drivers:
    /// - NVK/NAK: crashes on native exp(f64), log(f64)
    /// - RADV/ACO (AMD open-source): `fexp2` unimplemented for f64
    /// - NVIDIA proprietary (NVVM/PTXAS): fails to compile native f64
    ///   transcendentals (exp, log, pow, sin, cos, abs on f64). Observed on
    ///   Ada Lovelace (SM89) and Ampere (SM86). NVVM's PTXAS does not
    ///   implement double-precision transcendentals — it relies on libdevice
    ///   which SPIR-V cannot link. The fix: use our Cody-Waite + minimax
    ///   polynomial implementations from math_f64.wgsl.
    pub fn needs_f64_exp_log_workaround(&self) -> bool {
        self.is_nvk() || self.is_radv() || self.is_nvidia_proprietary()
    }

    /// Check if this device is NVIDIA Ada Lovelace (RTX 40xx) on the proprietary driver.
    ///
    /// This combination has broken f64 transcendentals in NVVM PTXAS.
    pub fn is_nvidia_ada_lovelace(&self) -> bool {
        let name = self.adapter_info.name.to_lowercase();
        let is_ada = name.contains("rtx 40") || name.contains("rtx40") || name.contains("l40");
        is_ada && self.is_nvidia_proprietary()
    }

    /// Probe and cache whether this device supports native f64 exp/log.
    ///
    /// This is the deep-debt evolution of `needs_f64_exp_log_workaround()`:
    /// instead of name-matching, it dispatches a tiny shader and verifies
    /// the result empirically. First call is async; subsequent calls use cache.
    pub async fn probe_f64_exp_capable(&self) -> bool {
        crate::device::probe::probe_f64_exp_capable(self).await
    }

    /// Check if this device uses a proprietary NVIDIA driver
    pub fn is_nvidia_proprietary(&self) -> bool {
        let name = self.adapter_info.name.to_lowercase();
        let driver = self.adapter_info.driver.to_lowercase();
        (name.contains("nvidia")
            || name.contains("geforce")
            || name.contains("rtx")
            || name.contains("gtx"))
            && !self.is_nvk()
            && !driver.contains("mesa")
    }
}
