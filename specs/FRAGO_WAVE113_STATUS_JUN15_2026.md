# FRAGO: barraCuda Wave 113 — Status for Overwatch

**Date**: 2026-06-15
**From**: barraCuda team (strandGate)
**To**: primalSpring (eastGate overwatch)
**Status**: All code obligations met. riboCipher shipped. Binary rebuilt. Awaiting deployment restart.

---

## COMPLETED (Waves 109–113 + Deep Debt)

| ID | Deliverable | Commit | Notes |
|----|-------------|--------|-------|
| W113-RIBO | riboCipher `[0xEC, 0x01]` acceptance | `97a69d8f` | `strip_ribocipher` via `BufReader::fill_buf()` — UDS + TCP |
| W113-EVOLVE | MSRV 1.92, let-chain modernization | `8a09bd26` | 83 collapsible_if, 24 suboptimal_flops, +22 tests |
| W109-STARTUP | guideStone `--bind-mode` / `PRIMAL_BIND_MODE` | `5f0e55e5` | Standard envelope compliant |
| W107-SOCKET | Socket co-location + `method.describe` | prev | `ProtectSystem=strict` compatible |
| W107-DEBT | Zero files >800L, zero unsafe, zero mocks | prev | Composition-ready |

---

## OPERATIONAL BLOCKER: Deployment Restart

**Problem**: primalSpring probe shows `barracuda | ❌ timeout` — the running binary on strandGate is the pre-riboCipher version.

**Fix**: Restart barracuda service with the new binary:
```bash
# Binary location (just rebuilt):
/home/strandgate/Development/ecoPrimals/primals/barraCuda/target/release/barracuda

# Standard startup:
barracuda server --bind-mode $PRIMAL_BIND_MODE --port $PORT
```

Once restarted, barracuda will respond to riboCipher-prefixed health probes and transition to `✅` in both signal and health columns.

**No code work required.** This is an ops restart.

---

## RESOLVED SINCE LAST FRAGO (Wave 93)

| Previous Issue | Resolution |
|---------------|------------|
| BTSP discovery tests env-dependent | Tests hardened with environment guards (Jun 15) — skip gracefully on hot hosts |
| GPU test parallelism SIGSEGV | Known, documented. nextest recommended. Not a code bug. |
| 2-Gate mesh proof | Federation operational (4-gate collective). No longer a blocker. |
| AMD RDNA 2 precision edge cases | Documented in AAR. Non-blocking (1-2 ULP). |

---

## REMAINING ISSUES (Informational)

### 1. GPU Test Parallelism — wgpu SIGSEGV (P3, infra, carry)

**Same as previous FRAGO.** Running `cargo test --workspace --lib` with default parallelism causes SIGSEGV in 50/3903 GPU tests due to Mesa llvmpipe resource contention. All IPC tests (593) pass clean. All tests pass individually or with nextest.

**Recommendation**: Ecosystem-wide `cargo-nextest` for GPU-bearing primals.

---

### 2. Probe Table Discrepancy (P2, ops)

primalSpring shows `barracuda | ❌ timeout` but the code is shipped and binary is built. This is a deploy-lag issue, not a code gap. Upstream should re-probe after binary restart on strandGate.

---

## BUILD VERIFICATION

```
$ cargo clippy --workspace --all-targets -- -D warnings
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.56s
    (zero warnings)

$ cargo test -p barracuda-core --lib ipc::
    test result: ok. 593 passed; 0 failed; 0 ignored

$ cargo test -p barracuda-core --test transport_config --test btsp_discovery
    test result: ok. 13 passed; 0 failed; 0 ignored

$ cargo build --release -p barracuda-core
    Finished `release` profile [optimized] target(s) in 1m 56s
    (barracuda: 5.5MB, ELF x86-64, stripped)

$ target/release/barracuda version
    barraCuda 0.4.0 | MSRV: 1.92 | Arch: x86_64 | OS: linux
```

---

## METRICS

| Metric | Value |
|--------|-------|
| Test count | 4,970+ |
| Clippy | zero warnings (pedantic + nursery) |
| Files >800L | 0 (production) |
| Unsafe code | 0 (`#![forbid(unsafe_code)]` on 3/4 crates) |
| Production unwrap | 0 |
| TODOs/FIXMEs | 0 |
| Dependencies | 100% pure Rust |
| JSON-RPC methods | 97 |
| MSRV | 1.92 |
| Edition | 2024 |

---

*"Code shipped. Binary built. Awaiting restart. Mountain clear."*
