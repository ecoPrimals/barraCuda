#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# barraCuda JSON-RPC Server Showcase
#
# Starts barraCuda as an IPC server, exercises JSON-RPC 2.0 methods,
# and demonstrates the primal protocol.

set -euo pipefail

BARRACUDA_BIN="${BARRACUDA_BIN:-../../../target/release/barracuda}"
BIND_ADDR="127.0.0.1:0"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  barraCuda — JSON-RPC Server Showcase                       ║"
echo "║  IPC protocol demonstration                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Check binary exists
if [ ! -f "$BARRACUDA_BIN" ]; then
    echo -e "${RED}Binary not found: $BARRACUDA_BIN${NC}"
    echo "Build first: cd ../../.. && cargo build --release -p barracuda-core"
    exit 1
fi

# Start server in background
echo "─── Starting barraCuda server ────────────────────────────────"
echo
$BARRACUDA_BIN server --bind "$BIND_ADDR" --no-unix &
SERVER_PID=$!
sleep 2

# Find the actual bound address from the discovery file
DISCOVERY_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ecoPrimals"
DISCOVERY_FILE="$DISCOVERY_DIR/barracuda-core.json"

if [ -f "$DISCOVERY_FILE" ]; then
    ADDR=$(grep -o '"address":"[^"]*"' "$DISCOVERY_FILE" | head -1 | cut -d'"' -f4)
    echo -e "  ${GREEN}Server running at $ADDR (PID $SERVER_PID)${NC}"
else
    echo -e "  ${RED}Discovery file not found, using fallback${NC}"
    ADDR="$BIND_ADDR"
fi
echo

cleanup() {
    echo
    echo "─── Shutting down server (PID $SERVER_PID) ───────────────────"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo -e "  ${GREEN}Server stopped${NC}"
}
trap cleanup EXIT

# Exercise JSON-RPC methods
call_method() {
    local method="$1"
    local params="${2:-{}}"
    echo -e "  ${CYAN}$method${NC}"
    $BARRACUDA_BIN client "$method" --params "$params" --addr "$ADDR" 2>/dev/null || echo "    (method returned error or server unreachable)"
    echo
}

echo "─── Exercising JSON-RPC 2.0 Methods ─────────────────────────"
echo

call_method "barracuda.primal.info"
call_method "barracuda.primal.capabilities"
call_method "barracuda.device.list"
call_method "barracuda.health.check"
call_method "barracuda.tolerances.get" '{"name":"default"}'
call_method "barracuda.validate.gpu_stack"

echo "─── Summary ──────────────────────────────────────────────────"
echo
echo "  Protocol:  JSON-RPC 2.0 (notification-compliant)"
echo "  Transport: TCP ($ADDR)"
echo "  Methods:   6 exercised from barracuda.{domain}.{operation}"
echo "  Discovery: $DISCOVERY_FILE"
echo
