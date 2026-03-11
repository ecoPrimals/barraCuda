#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# barraCuda Doctor + Validate Showcase
#
# Demonstrates self-diagnostic commands for production health checking.

set -euo pipefail

BARRACUDA_BIN="${BARRACUDA_BIN:-../../../target/release/barracuda}"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  barraCuda — Doctor + Validate Showcase                     ║"
echo "║  Self-diagnostic and GPU validation canary                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

if [ ! -f "$BARRACUDA_BIN" ]; then
    echo -e "${RED}Binary not found: $BARRACUDA_BIN${NC}"
    echo "Build first: cd ../../.. && cargo build --release -p barracuda-core"
    exit 1
fi

# Version info
echo "─── barracuda version ────────────────────────────────────────"
echo
$BARRACUDA_BIN version
echo

# Doctor (health diagnostics)
echo "─── barracuda doctor ─────────────────────────────────────────"
echo
$BARRACUDA_BIN doctor
echo

# Validate (GPU canary)
echo "─── barracuda validate ───────────────────────────────────────"
echo
$BARRACUDA_BIN validate && {
    echo
    echo -e "  ${GREEN}GPU validation passed${NC}"
} || {
    echo
    echo -e "  ${RED}GPU validation failed (expected on software rendering)${NC}"
}
echo

echo "─── Summary ──────────────────────────────────────────────────"
echo
echo "  doctor:   Reports GPU health, driver info, feature support"
echo "  validate: Dispatches test shaders, verifies correctness"
echo "  version:  Build info for reproducibility"
echo
echo "  These commands run in CI and on genomeBin deployment."
echo
