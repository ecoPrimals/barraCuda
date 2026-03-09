#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# toadStool Hardware Discovery Showcase
#
# Demonstrates toadStool hardware inventory feeding barraCuda GPU selection.
# Gracefully degrades to barraCuda local discovery when toadStool is absent.

set -euo pipefail

BARRACUDA_BIN="${BARRACUDA_BIN:-../../../target/release/barracuda}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  barraCuda — toadStool Hardware Discovery Showcase          ║"
echo "║  Cross-primal hardware inventory for GPU selection          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# --- Phase 1: Discover toadStool ---
echo "─── Phase 1: Discover toadStool ──────────────────────────────"
echo

DISCOVERY_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ecoPrimals"
TOADSTOOL_FOUND=false

if [ -d "$DISCOVERY_DIR" ]; then
    echo "  Scanning: $DISCOVERY_DIR/"
    echo
    for manifest in "$DISCOVERY_DIR"/*.json; do
        [ -f "$manifest" ] || continue
        name=$(basename "$manifest")
        echo -e "  Found: ${CYAN}$name${NC}"

        # Check for hardware_discovery capability
        if grep -q "hardware_discovery\|gpu_management" "$manifest" 2>/dev/null; then
            echo -e "    ${GREEN}Has hardware discovery capability${NC}"
            TOADSTOOL_FOUND=true
        fi
    done
else
    echo "  Discovery directory not found: $DISCOVERY_DIR"
fi
echo

# --- Phase 2: Query or Fallback ---
echo "─── Phase 2: Hardware Inventory ──────────────────────────────"
echo

if [ "$TOADSTOOL_FOUND" = true ]; then
    echo -e "  ${GREEN}toadStool available — querying hardware inventory...${NC}"
    echo
    # In production, barraCuda would call toadStool's IPC methods.
    # For the showcase, we demonstrate the discovery pattern.
    echo "  toadStool provides:"
    echo "    - GPU enumeration (NVIDIA, AMD, Intel)"
    echo "    - NPU detection (Akida)"
    echo "    - CPU capability profiling"
    echo "    - Hardware health monitoring"
else
    echo -e "  ${YELLOW}toadStool not running — using barraCuda local discovery${NC}"
    echo
    echo "  barraCuda discovers GPUs directly via wgpu:"
fi
echo

# --- Phase 3: barraCuda Local Discovery (always works) ---
echo "─── Phase 3: barraCuda Local GPU Discovery ───────────────────"
echo

if [ -f "$BARRACUDA_BIN" ]; then
    $BARRACUDA_BIN doctor
else
    echo "  Binary not built. Run: cd ../../.. && cargo build --release -p barracuda-core"
fi
echo

# --- Summary ---
echo "─── Summary ──────────────────────────────────────────────────"
echo
echo "  Compute triangle:"
echo "    toadStool → hardware discovery + GPU management"
echo "    barraCuda → math engine (716+ WGSL shaders)"
echo "    coralReef → sovereign shader compilation"
echo
echo "  When toadStool is available:"
echo "    barraCuda receives richer hardware profiles (NPU, multi-GPU)"
echo "    toadStool manages GPU health and load balancing"
echo
echo "  When toadStool is absent:"
echo "    barraCuda discovers GPUs locally via wgpu"
echo "    Full math capability preserved — only orchestration is reduced"
echo
