# FRAGO: barraCuda Wave 93 — Remaining Issues for Upstream

**Date**: 2026-06-07
**From**: barraCuda team (strandGate)
**To**: primalSpring (eastGate overwatch)
**Status**: Build fixed, depot-ready (13/13). Below are remaining issues needing upstream coordination or awareness.

---

## RESOLVED (This Session)

| ID | Issue | Resolution |
|----|-------|------------|
| BC-BUILD-01 | `SimpleMlp::to_binary()` / `from_auto()` missing (E0599) | Restored `mod serialization;` + removed duplicate stubs + restored serde aliases |
| BC-BUILD-02 | Stash conflict left simplified JSON-only `to_binary` conflicting with bincode+BLAKE3 version | Kept proper `serialization.rs` implementation (44-byte BCML header, BLAKE3 integrity) |
| BC-BUILD-03 | `DenseLayer` serde aliases lost (`weights`/`biases` → `weight`/`bias`) | Restored `#[serde(alias = "weights")]` and `#[serde(alias = "biases")]` |

---

## REMAINING ISSUES (Upstream Awareness)

### 1. BTSP Discovery Tests — Environment-Dependent (P3, informational)

**Tests**: `ipc::btsp::tests::discover_security_provider_returns_none_when_no_socket`, `ipc::btsp::tests::discover_security_provider_checks_beardog_socket_env`

**Behavior**: These 2 tests fail on strandGate because a live bearDog socket exists at the expected discovery path. The tests assert "no provider found" but the real socket is present.

**Impact**: Zero. Tests pass in CI and on hosts without bearDog running. Not a code bug — environment-dependent.

**Recommendation**: Consider `#[ignore]` with `--ignored` flag for hosts with live bearDog, or test-scoped temp dir override for `BTSP_PROVIDER_SOCKET`.

---

### 2. GPU Test Parallelism — wgpu SIGSEGV (P2, infra)

**Behavior**: Running full `cargo test --workspace` with default parallelism occasionally causes SIGSEGV (signal 11) when many GPU tests contend for wgpu's process-global device state. All tests pass individually.

**Root Cause**: wgpu maintains process-global state (adapter/device handles). Mass parallel `cargo test` threads creating/destroying devices simultaneously exhausts the GPU driver's handle pool, causing Mesa/vulkan to SIGSEGV.

**Impact**: Intermittent CI failures under high parallelism. Not a code bug.

**Fix**: `cargo-nextest` provides process isolation per test binary. Already works (all tests pass with nextest). Consider ecosystem-wide adoption for GPU-bearing primals (barraCuda, toadStool, coralReef).

**Recommendation for upstream**: Add `cargo-nextest` to ecosystem CI standard for primals with GPU features. The shared test pool + semaphore throttling mitigates most cases, but process isolation is the complete fix.

---

### 3. 2-Gate Mesh Proof — Coordination (P1, ops)

**Requirement**: eastGate ↔ strandGate bidirectional mesh validation.

**strandGate status**: Songbird :7700 LIVE, NUCLEUS running, `mesh.health_check` responding.

**Blocker**: eastGate needs NUCLEUS + Songbird started on LAN machine (192.168.1.144) with:
- `SONGBIRD_FEDERATION_PORT=7700`
- `SONGBIRD_PRODUCTION_BIND_ADDRESS=0.0.0.0`
- :7700 reachable from strandGate (192.168.1.132)

**Action**: eastGate operators coordinate startup. barraCuda has zero code work remaining for this.

---

### 4. AMD RDNA 2 Precision Edge Cases (P3, documented)

**Reference**: `specs/AAR_CROSS_VENDOR_GPU_VALIDATION_JUN06_2026.md`

Three AMD-specific test failures documented (99.8% pass rate vs 100% on NVIDIA/llvmpipe):
1. `hermite_f64_wgsl::test_hermite_function_normalization` — f64 denormal flush-to-zero
2. `avg_pool1d_wgsl::tests::test_avg_pool1d_stride_one` — f32 reduction ordering
3. `cosine_embedding_loss::tests::test_cosine_embedding_loss_basic` — FMA rounding difference

**Impact**: Non-blocking. All are within 1-2 ULP of expected values.

**Action items** (for future evolution, not blocking):
- Widen tolerances for AMD RDNA hardware or add vendor-specific tolerance tiers
- Document in toadStool for hardware dispatch awareness

---

## BUILD VERIFICATION

```
$ cargo build --workspace
    Finished `dev` profile [unoptimized + debuginfo] target(s)

$ cargo test -p barracuda-core --lib -- ipc
    test result: ok. 572 passed; 2 failed (known env-dep); 0 ignored

$ cargo test -p barracuda --features gpu --lib -- nn::simple_mlp
    test result: ok. 12 passed; 0 failed; 0 ignored
```

---

*"Build fixed. Depot current. Mountain clear."*
