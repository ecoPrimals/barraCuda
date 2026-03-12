#!/usr/bin/env bash
# barraCuda tiered test dispatch — GPU optimization strategies applied to testing
#
# GPU ↔ Test Strategy Mapping:
#   Workgroup sizing   → nextest thread count (match hardware capacity)
#   Pipeline cache     → test_pool device caching (amortize creation)
#   Dispatch semaphore → nextest profiles (bounded concurrency)
#   Exponential backoff → device creation retries (handle oversubscription)
#   Occupancy control  → tiered dispatch (fast first, heavy last)
#
# Tiers:
#   1. STATIC — clippy + compile (catches 80% of issues in seconds)
#   2. CORE  — barracuda-core lib tests (IPC, lifecycle, RPC)
#   3. TARGETED — changed-module tests only (sovereign, tolerances, etc.)
#   4. FULL  — all 3,688+ tests via nextest (bounded parallelism)
#   5. CORAL — coralReef cross-primal validation (shader compilation probes)
#   6. GPU   — hardware workload tests (BARRACUDA_TEST_BACKEND=gpu)
#
# Usage:
#   ./scripts/test-tiered.sh           # Tiers 1-3 (fast iteration)
#   ./scripts/test-tiered.sh full      # Tiers 1-4
#   ./scripts/test-tiered.sh coralreef # Tiers 1-5 (with coralReef probe)
#   ./scripts/test-tiered.sh gpu       # Tiers 1-6
#   ./scripts/test-tiered.sh quick     # Tier 1 only
#   ./scripts/test-tiered.sh stress    # All tiers + oversubscription stress test
#   ./scripts/test-tiered.sh <filter>  # Targeted: test names matching <filter>

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

tier()  { echo -e "\n${CYAN}━━━ Tier $1: $2 ━━━${NC}"; }
ok()    { echo -e "  ${GREEN}✓ $1${NC}"; }
skip()  { echo -e "  ${YELLOW}⊘ $1${NC}"; }
fail()  { echo -e "  ${RED}✗ $1${NC}"; }
elapsed() { echo "$(( $(date +%s%3N) - $1 ))ms"; }

MODE="${1:-default}"
NEXTEST="cargo nextest run"

# Handle direct filter mode (e.g., ./test-tiered.sh sovereign)
case "$MODE" in
    default|full|coralreef|gpu|quick|stress) ;;
    *)
        tier 0 "Targeted: $MODE"
        T=$(date +%s%3N)
        $NEXTEST -p barracuda --lib -E "test($MODE)" --profile default
        ok "$MODE tests passed ($(elapsed $T))"
        exit 0
        ;;
esac

# ─── Tier 1: Static analysis ──────────────────────────────────────
tier 1 "Static analysis (clippy + compile)"
T=$(date +%s%3N)
cargo clippy --workspace --all-targets --all-features -- -D warnings 2>&1 | tail -1
ok "Clippy clean ($(elapsed $T))"

[ "$MODE" = "quick" ] && { echo -e "\n${GREEN}Quick check complete.${NC}"; exit 0; }

# ─── Tier 2: Core library tests ───────────────────────────────────
tier 2 "Core library (barracuda-core: IPC, lifecycle, RPC)"
T=$(date +%s%3N)
$NEXTEST -p barracuda-core --lib --profile default
ok "barracuda-core: 50 tests ($(elapsed $T))"

# ─── Tier 3: Targeted module tests ────────────────────────────────
tier 3 "Targeted modules (changed-code validation)"
T=$(date +%s%3N)

$NEXTEST -p barracuda --lib -E 'test(sovereign)' --profile default 2>&1 | tail -1
ok "Sovereign compiler: df64_rewrite + spv_emit + dead_expr"

$NEXTEST -p barracuda --lib -E 'test(tolerances)' --profile default 2>&1 | tail -1
ok "Tolerances: tier ordering + const guards"

$NEXTEST -p barracuda --lib -E 'test(ncbi_cache)' --profile default 2>&1 | tail -1
ok "ncbi_cache: XDG paths (pure std)"

ok "Targeted tier complete ($(elapsed $T))"

[ "$MODE" = "default" ] && {
    echo -e "\n${GREEN}Tiers 1-3 complete. Use 'full' for all unit tests, 'gpu' for hardware tests.${NC}"
    exit 0
}

# ─── Tier 4: Full unit tests (nextest, bounded parallelism) ───────
tier 4 "Full unit tests (3,688+ via nextest, 16 threads)"
T=$(date +%s%3N)

$NEXTEST -p barracuda --lib --profile default
ok "barracuda lib: all tests ($(elapsed $T))"

T2=$(date +%s%3N)
$NEXTEST -p barracuda --tests --profile default
ok "barracuda integration: all tests ($(elapsed $T2))"

[ "$MODE" = "full" ] && { echo -e "\n${GREEN}Full suite complete.${NC}"; exit 0; }

# ─── Tier 5: coralReef cross-primal validation ────────────────────
if [ "$MODE" = "coralreef" ] || [ "$MODE" = "gpu" ] || [ "$MODE" = "stress" ]; then
    tier 5 "coralReef cross-primal validation"
    T=$(date +%s%3N)

    # Probe coralReef availability via its XDG runtime manifest
    CORAL_MANIFEST=$(find "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/ecoPrimals" -name 'coralReef*.json' 2>/dev/null | head -1)
    if [ -n "$CORAL_MANIFEST" ]; then
        ok "coralReef discovered at $CORAL_MANIFEST"

        # Run sovereign compiler tests that exercise the coralReef IPC path
        $NEXTEST -p barracuda --lib -E 'test(coral)' --profile default 2>&1 | tail -1
        ok "coralReef shader compilation tests ($(elapsed $T))"

        $NEXTEST -p barracuda --lib -E 'test(sovereign)' --profile default 2>&1 | tail -1
        ok "Sovereign compiler + coralReef validation ($(elapsed $T))"
    else
        skip "coralReef not running — cross-primal tests skipped ($(elapsed $T))"
    fi

    [ "$MODE" = "coralreef" ] && { echo -e "\n${GREEN}Tiers 1-5 (coralReef) complete.${NC}"; exit 0; }
fi

# ─── Tier 6: GPU workload tests ───────────────────────────────────
if [ "$MODE" = "gpu" ] || [ "$MODE" = "stress" ]; then
    tier 6 "GPU workload (BARRACUDA_TEST_BACKEND=gpu)"
    T=$(date +%s%3N)
    BARRACUDA_TEST_BACKEND=gpu $NEXTEST --workspace --profile gpu
    ok "GPU workload ($(elapsed $T))"
fi

# ─── Stress tier: intentional oversubscription ─────────────────────
if [ "$MODE" = "stress" ]; then
    tier 7 "Stress (128 threads — tests device creation backoff)"
    T=$(date +%s%3N)
    $NEXTEST -p barracuda --lib --profile stress --no-fail-fast 2>&1 | tail -3
    ok "Stress test ($(elapsed $T))"
fi

echo -e "\n${GREEN}All requested tiers complete.${NC}"
