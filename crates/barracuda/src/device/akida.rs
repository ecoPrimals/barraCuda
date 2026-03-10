// SPDX-License-Identifier: AGPL-3.0-only
// ! Akida NPU Backend for barraCuda
//!
//! Provides Akida-specific acceleration for neuromorphic operations
//! while maintaining compatibility with the universal barraCuda API.
//!
//! **Architecture**:
//! - Detects available Akida boards
//! - Translates select operations to Akida-native format
//! - Falls back to wgpu for non-neuromorphic operations
//! - Maintains zero hardcoding principle

use crate::error::Result;
use std::path::PathBuf;

/// Akida board information
#[derive(Debug, Clone)]
pub struct AkidaBoard {
    /// Board index (0-based)
    pub index: usize,

    /// `PCIe` address (e.g., "a1:00.0")
    pub pcie_address: String,

    /// Device path (e.g., "/dev/akida0")
    pub device_path: PathBuf,

    /// Chip name (e.g., "Akida AKD1000")
    pub chip_name: String,

    /// Number of NPUs
    pub npu_count: usize,

    /// Available memory in bytes
    pub memory_bytes: usize,

    /// Current power consumption (watts)
    pub power_watts: f64,

    /// Current temperature (celsius)
    pub temperature_celsius: f64,

    /// `PCIe` generation (1-4)
    pub pcie_generation: u8,

    /// `PCIe` lane count
    pub pcie_lanes: u8,

    /// Health status
    pub health: BoardHealth,
}

/// Board health status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardHealth {
    /// Operating normally
    Healthy,

    /// Operating with warnings (high temp, etc.)
    Warning,

    /// Not responding or critical error
    Critical,

    /// Not yet queried
    Unknown,
}

/// Akida backend capabilities
#[derive(Debug, Clone)]
pub struct AkidaCapabilities {
    /// Boards detected
    pub boards: Vec<AkidaBoard>,

    /// Total NPUs across all boards
    pub total_npus: usize,

    /// Total memory across all boards
    pub total_memory_bytes: usize,

    /// SDK version (if available)
    pub sdk_version: Option<String>,
}

/// Detect available Akida boards
///
/// **Deep Debt**: Runtime discovery, zero hardcoding
///
/// # Errors
///
/// Returns [`Err`] if `PCIe` scanning fails or board query fails (e.g. permission
/// denied, driver unavailable).
pub fn detect_akida_boards() -> Result<AkidaCapabilities> {
    tracing::info!("Detecting Akida NPU boards...");

    // Scan PCIe bus for BrainChip devices
    let pcie_devices = scan_pcie_for_akida()?;

    if pcie_devices.is_empty() {
        tracing::info!("No Akida boards detected");
        return Ok(AkidaCapabilities {
            boards: Vec::new(),
            total_npus: 0,
            total_memory_bytes: 0,
            sdk_version: None,
        });
    }

    tracing::info!("Found {} Akida board(s)", pcie_devices.len());

    // Query each board
    let boards: Vec<_> = pcie_devices
        .iter()
        .enumerate()
        .filter_map(|(index, device)| match query_board_info(device, index) {
            Ok(board) => {
                tracing::info!(
                    "  Board {}: {} at {} ({} NPUs, {:.1}W, {:.1}°C)",
                    index,
                    board.chip_name,
                    board.pcie_address,
                    board.npu_count,
                    board.power_watts,
                    board.temperature_celsius
                );
                Some(board)
            }
            Err(e) => {
                tracing::warn!("Failed to query board {}: {}", index, e);
                None
            }
        })
        .collect();

    // Calculate totals
    let total_npus = boards.iter().map(|b| b.npu_count).sum();
    let total_memory_bytes = boards.iter().map(|b| b.memory_bytes).sum();

    // Try to detect SDK version
    let sdk_version = detect_akida_sdk_version();

    Ok(AkidaCapabilities {
        boards,
        total_npus,
        total_memory_bytes,
        sdk_version,
    })
}

/// Chip-specific specs resolved from `PCIe` device ID at runtime.
struct AkidaChipSpecs {
    name: &'static str,
    npu_count: usize,
    memory_bytes: usize,
}

impl AkidaChipSpecs {
    fn from_device_id(device_id: u16) -> Self {
        match device_id {
            0x1000 => Self {
                name: "Akida AKD1000",
                npu_count: 80,
                memory_bytes: 10 * 1024 * 1024,
            },
            0x1500 => Self {
                name: "Akida AKD1500",
                npu_count: 128,
                memory_bytes: 32 * 1024 * 1024,
            },
            _ => {
                tracing::warn!(
                    device_id,
                    "Unknown Akida device ID, using conservative defaults"
                );
                Self {
                    name: "Akida (unknown)",
                    npu_count: 1,
                    memory_bytes: 1024 * 1024,
                }
            }
        }
    }
}

/// `PCIe` device info
#[derive(Debug, Clone)]
struct PcieDevice {
    address: String,
    #[expect(dead_code, reason = "populated during PCIe scan, used for diagnostics")]
    vendor_id: u16,
    device_id: u16,
}

/// Scan `PCIe` bus for `BrainChip` Akida devices
fn scan_pcie_for_akida() -> Result<Vec<PcieDevice>> {
    // Scan /sys/bus/pci/devices for BrainChip vendor ID (0x1e7c)
    let pci_dir = "/sys/bus/pci/devices";

    let devices = if let Ok(entries) = std::fs::read_dir(pci_dir) {
        entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let address = entry.file_name().to_string_lossy().to_string();

                // Read vendor and device IDs
                let vendor_path = path.join("vendor");
                let device_path = path.join("device");

                let (vendor_str, device_str) = (
                    std::fs::read_to_string(&vendor_path),
                    std::fs::read_to_string(&device_path),
                );

                let (Ok(vendor_str), Ok(device_str)) = (vendor_str, device_str) else {
                    return None;
                };

                let Ok(vendor_id) =
                    u16::from_str_radix(vendor_str.trim().trim_start_matches("0x"), 16)
                else {
                    return None;
                };
                let Ok(device_id) =
                    u16::from_str_radix(device_str.trim().trim_start_matches("0x"), 16)
                else {
                    return None;
                };

                // BrainChip vendor ID is 0x1e7c
                // Akida AKD1000 device ID is 0x1000
                if vendor_id == 0x1e7c {
                    tracing::debug!(
                        "Found BrainChip device at {}: {:04x}:{:04x}",
                        address,
                        vendor_id,
                        device_id
                    );

                    Some(PcieDevice {
                        address,
                        vendor_id,
                        device_id,
                    })
                } else {
                    None
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(devices)
}

/// Query board information
fn query_board_info(device: &PcieDevice, index: usize) -> Result<AkidaBoard> {
    let device_path = PathBuf::from(format!("/dev/akida{index}"));

    let (pcie_gen, pcie_lanes) = query_pcie_link_info(&device.address)?;

    // Query real power consumption from hwmon
    let power_watts = query_power_consumption(&device.address);

    // Query real temperature from hwmon
    let temperature_celsius = query_temperature(&device.address);

    let specs = AkidaChipSpecs::from_device_id(device.device_id);

    let board = AkidaBoard {
        index,
        pcie_address: device.address.clone(),
        device_path,
        chip_name: specs.name.to_string(),
        npu_count: specs.npu_count,
        memory_bytes: specs.memory_bytes,
        power_watts,
        temperature_celsius,
        pcie_generation: pcie_gen,
        pcie_lanes,
        health: check_board_health(&device.address)?,
    };

    Ok(board)
}

/// Query `PCIe` link status
fn query_pcie_link_info(address: &str) -> Result<(u8, u8)> {
    use std::fs;

    let base_path = format!("/sys/bus/pci/devices/{address}");

    // Read current link speed and width
    let speed_path = format!("{base_path}/current_link_speed");
    let width_path = format!("{base_path}/current_link_width");

    let generation = if let Ok(speed) = fs::read_to_string(&speed_path) {
        parse_pcie_speed(&speed)
    } else {
        2 // Default Gen2
    };

    let lanes = if let Ok(width) = fs::read_to_string(&width_path) {
        width.trim().parse().unwrap_or_else(|_| {
            tracing::debug!(address, "unparseable PCIe width, defaulting to x4");
            4
        })
    } else {
        4
    };

    Ok((generation, lanes))
}

/// Parse `PCIe` speed to generation
fn parse_pcie_speed(speed: &str) -> u8 {
    if speed.contains("2.5") {
        1 // Gen1: 2.5 GT/s
    } else if speed.contains("5.0") || speed.contains("5 GT") {
        2 // Gen2: 5.0 GT/s
    } else if speed.contains("8.0") || speed.contains("8 GT") {
        3 // Gen3: 8.0 GT/s
    } else if speed.contains("16.0") || speed.contains("16 GT") {
        4 // Gen4: 16.0 GT/s
    } else {
        2 // Default Gen2
    }
}

/// Query power consumption from hwmon
/// Deep Debt: Real hardware monitoring, no estimates!
fn query_power_consumption(pcie_address: &str) -> f64 {
    use std::fs;

    // Search for hwmon directory
    let hwmon_base = format!("/sys/bus/pci/devices/{pcie_address}/hwmon");

    if let Ok(entries) = fs::read_dir(&hwmon_base) {
        for entry in entries.flatten() {
            let hwmon_path = entry.path();
            let power_input_path = hwmon_path.join("power1_input");

            // power1_input is in microwatts
            if let Ok(power_str) = fs::read_to_string(&power_input_path) {
                if let Ok(power_uw) = power_str.trim().parse::<f64>() {
                    let power_watts = power_uw / 1_000_000.0; // Convert µW to W
                    tracing::debug!(
                        "Akida {}: Measured power = {:.3}W",
                        pcie_address,
                        power_watts
                    );
                    return power_watts;
                }
            }
        }
    }

    // Fallback: Use Akida AKD1000 typical power (0.5-2W range)
    // But log that we're using fallback
    tracing::warn!(
        "Akida {}: hwmon not available, using typical power estimate",
        pcie_address
    );
    1.0 // Typical idle power
}

/// Query temperature from hwmon
/// Deep Debt: Real hardware monitoring, no estimates!
fn query_temperature(pcie_address: &str) -> f64 {
    use std::fs;

    // Search for hwmon directory
    let hwmon_base = format!("/sys/bus/pci/devices/{pcie_address}/hwmon");

    if let Ok(entries) = fs::read_dir(&hwmon_base) {
        for entry in entries.flatten() {
            let hwmon_path = entry.path();
            let temp_input_path = hwmon_path.join("temp1_input");

            // temp1_input is in millidegrees celsius
            if let Ok(temp_str) = fs::read_to_string(&temp_input_path) {
                if let Ok(temp_mdeg) = temp_str.trim().parse::<f64>() {
                    let temp_celsius = temp_mdeg / 1000.0; // Convert millidegrees to degrees
                    tracing::debug!(
                        "Akida {}: Measured temperature = {:.1}°C",
                        pcie_address,
                        temp_celsius
                    );
                    return temp_celsius;
                }
            }
        }
    }

    // Fallback: Use Akida AKD1000 typical operating temperature
    // But log that we're using fallback
    tracing::warn!(
        "Akida {}: hwmon not available, using typical temperature estimate",
        pcie_address
    );
    40.0 // Typical operating temperature
}

/// Check board health
fn check_board_health(address: &str) -> Result<BoardHealth> {
    // Check if device is accessible
    let base_path = format!("/sys/bus/pci/devices/{address}");

    if std::fs::metadata(&base_path).is_ok() {
        // Device exists and is accessible
        Ok(BoardHealth::Healthy)
    } else {
        Ok(BoardHealth::Unknown)
    }
}

/// Well-known system locations for the Akida SDK — tried only after
/// environment variables (`AKIDA_SDK_DIR`, `XDG_DATA_HOME`) are exhausted.
pub(crate) const AKIDA_SDK_SYSTEM_DIRS: &[&str] =
    &["/opt/akida", "/usr/local/akida", "/usr/share/akida"];

/// Detect Akida SDK version via capability-based discovery.
///
/// Resolution chain (first match wins):
/// 1. `AKIDA_SDK_DIR` environment variable → `$AKIDA_SDK_DIR/version`
/// 2. XDG data dirs → `$XDG_DATA_HOME/akida/version`
/// 3. System paths from [`AKIDA_SDK_SYSTEM_DIRS`]
fn detect_akida_sdk_version() -> Option<String> {
    let mut search_paths = Vec::new();

    if let Ok(sdk_dir) = std::env::var("AKIDA_SDK_DIR") {
        search_paths.push(format!("{sdk_dir}/version"));
    }

    if let Ok(data_home) = std::env::var("XDG_DATA_HOME") {
        search_paths.push(format!("{data_home}/akida/version"));
    } else if let Ok(home) = std::env::var("HOME") {
        search_paths.push(format!("{home}/.local/share/akida/version"));
    }

    for dir in AKIDA_SDK_SYSTEM_DIRS {
        search_paths.push(format!("{dir}/version"));
    }

    for path in &search_paths {
        if let Ok(version) = std::fs::read_to_string(path) {
            return Some(version.trim().to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_akida_detection() {
        let caps = detect_akida_boards().expect("detect_akida_boards should succeed on any system");
        println!("Detected {} Akida boards", caps.boards.len());

        for board in &caps.boards {
            println!("  Board {}: {}", board.index, board.chip_name);
            println!(
                "    PCIe: Gen{} x{}",
                board.pcie_generation, board.pcie_lanes
            );
            println!("    NPUs: {}", board.npu_count);
            println!("    Memory: {} MB", board.memory_bytes / (1024 * 1024));
        }
    }
}
