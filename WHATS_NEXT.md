# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-04-20.

---

## Recently Completed

- **Sprint 44c: Phase 45 Audit — CPU Tensor Fallback (Apr 20)**: CPU fallback for
  all 7 handle-based tensor ops (`tensor.create`, `matmul`, `add`, `scale`, `clamp`,
  `reduce`, `sigmoid`) via `CpuTensor` store — headless hosts no longer return
  "No GPU device available." IPC namespace guide added to wire contract (9 namespaces).
  Socket naming clarified (authoritative `math.sock` vs legacy `barracuda.sock`).
  12-axis deep debt scan clean. Resolves primalSpring Phase 45 gap #6.
- **Sprint 44: primalSpring Composition Audit (Apr 20)**: 6 new JSON-RPC methods
  wired (32→39): `stats.variance`, `stats.correlation`, `linalg.solve`,
  `linalg.eigenvalues`, `spectral.fft`, `spectral.power_spectrum` — unblocks
  Level 5 NUCLEUS certification for wetSpring, healthSpring, neuralSpring.
  `tensor.matmul_inline` convenience path (inline data, no handle round-trip).
  Fitts' law Shannon formula corrected (`log₂(D/W + 1)` per MacKenzie 1992).
  Response schema standardized (`"result"` key on all scalar methods).
  `stats.std_dev`/`stats.variance` convention documented (sample, N-1 denominator).
  Hick default verified (`log₂(N)`). Perlin3d origin verified (returns 0.0).
  2 new IPC domains: `linalg`, `spectral`.
- **Sprint 43b: Deep Debt Evolution (Apr 15)**: Smart WGSL refactoring
  (`math_f64.wgsl` 840→725L, fossils extracted to `math_f64_fossils.wgsl`).
  `asin_f64` evolved from fossil `sqrt_f64` to native `sqrt()`. biomeos namespace
  hardcoding evolved to env-overridable via `BIOMEOS_SOCKET_DIR`. HMAC expects
  evolved to `map_err` (zero `expect()` in crypto paths). 12-axis deep debt audit:
  all `.rs` under 800L, zero TODO/FIXME, zero mocks in production. Benchmark
  infrastructure assessed (Kokkos parity operational, no Python baselines in-tree).
- **Sprint 43: BTSP Phase 3, BufReader Fix & Gap Resolution (Apr 15)**: BTSP Phase 3
  stream encryption — `BtspCipher` + `BtspSession` + `BtspFrameReader`/`BtspFrameWriter`
  with ChaCha20-Poly1305/HMAC-SHA256/NULL cipher suites. Length-prefixed framing per
  BTSP spec. Transport integration on all accept loops. BufReader lifetime fix (single
  instance with `get_mut()`). plasma_dispersion verified.   18/18 neuralSpring shader absorption confirmed (per-shader audit table in
  `SPRING_ABSORPTION.md`). 4 new crypto deps (RustCrypto).
- **Sprint 42: Composition Elevation, LD-05 Fix & Deep Debt Evolution (Apr 12)**:
  **LD-05 fully resolved** — Phase 1: bind-before-discovery prevents phantom TCP endpoints.
  Phase 2: UDS mode no longer attempts TCP sidecar from `BARRACUDA_PORT` env var — only
  explicit `--port`/`--bind` CLI triggers TCP bind, eliminating co-deployment port collisions
  entirely. `serve_tarpc` gracefully degrades on `AddrInUse`. TCP-only fallback uses
  `try_bind_tcp`. Standardized all `tensor.*` IPC response schemas for primalSpring typed
  extractors. Implemented `tensor.batch.submit` — fused multi-op GPU pipeline over
  JSON-RPC. Smart refactored `sovereign_device.rs` and `transfer.rs`. `primal.device()`
  evolved to `Arc<WgpuDevice>`. Showcase hardcoding evolved to capability-based.
  **Deep debt cleanup**: `BatchError` typed error (zero `Result<T,String>`), `.expect()`
  eliminated with `let-else`, `with_device` constructors on 8 types, precision preambles
  extracted (722→409 lines), lanczos iterator evolution, 9 new tests.
  **Phase 3**: `invocation.rs` smart-refactored (754→445, memory ops → `memory.rs`).
  `wgpu_device/mod.rs` smart-refactored (729→518, submit/poll → `submission.rs`). All
  production files under 600 lines. `as usize` → `usize::try_from` in batch handler.
  36 new tests (math/stats/noise/rng/activation/batch validation). Pre-existing clippy
  debt resolved. **Phase 4**: LD-10 resolved — BTSP guard consumed first line from legacy
  JSON-RPC clients; `BtspOutcome::Degraded` now carries consumed line for replay.
  `dispatch_line` helper extracted (DRY). 32 IPC methods. **Phase 5**: `NagaExecError::Overflow`
  typed error for workgroup size overflow (checked_mul through u64). 15 new tensor IPC handler
  tests (tensor.add/scale/clamp/reduce/sigmoid — all 5 previously untested handlers covered).
  4,358 tests pass. **Phase 6**: 8 new coverage tests (identity_get tarpc, FHE degree overflow,
  has_sovereign_dispatch, compute_device, health_readiness after start, whitespace batch).
  Stale `clippy::unused_async` crate-level expectation removed. 4,366 tests pass.
  **Phase 7**: `data_arr.len() as u64` → safe `u64::try_from` (last uncovered truncation cast).
  Path-dep versions aligned (0.3.11/0.3.6 → 0.3.12). Visibility tightened: `REGISTERED_METHODS`,
  `normalize_method`, `provided_capability_groups` to `pub(crate)`. 2 new JSON-RPC FHE degree
  overflow tests. 4,368 tests pass. **Phase 8**: Duplicated `"127.0.0.1"` in binary replaced
  with `transport::DEFAULT_BIND_HOST`. Batch pre-validation elevated (scale/layer_norm/reshape
  parameter checks before device availability — `INVALID_PARAMS` not `INTERNAL_ERROR`). Dead
  `NagaExecError::NotCompute` removed. 3 new batch validation tests. 4,371 tests pass.
  **Phase 9 — Dead code removal + coverage**: 5 dead `WgpuDevice` functions removed,
  6 new GPU-free coverage tests. 4,377 tests pass.
  **Phase 10 — BC-09 Docker TCP bind**: `resolve_bind_host()` respects `BARRACUDA_IPC_HOST`.
  **Phase 11 — Runtime extraction + coverage**: `tokio_block_on` extracted from test_pool
  to `crate::runtime`. 14 new GPU-free type validation tests. 4,393 tests pass.
- **Sprint 41: BC-07 Full Wiring + BC-06 Docs + TensorSession Migration Guide (Apr 11)**:
  `Auto::new()` returns `DiscoveredDevice` enum with 3-tier fallback (wgpu GPU → wgpu CPU
  → SovereignDevice IPC → Err). `BarraCudaPrimal` stores `DiscoveredDevice`. `Auto::new_wgpu()`
  convenience added. BC-06 musl-static GPU constraint documented in README + CONTEXT. Full
  TensorSession migration guide published in BREAKING_CHANGES.md. 11-axis deep debt audit:
  clean bill — hardcoded primal names evolved to capability-based. 4,251 tests pass.
- **Sprint 40: primalSpring Gap Resolution & Deep Debt Overstep Cleanup (Apr 11)**:
  BC-07 partial (sovereign probe), BC-08 resolved (cpu-shader default-on), plasma_dispersion
  feature-gate fixed, TensorSession→BatchGuard rename, validation_harness typed errors, 670+
  println/eprintln removed, FHE tests typed, health ops tracing::warn. 68 files changed.
- **Sprint 39: primalSpring Audit Remediation (Apr 10)**: BTSP Phase 2 full handshake —
  `guard_connection()` evolved to 6-step X25519+HMAC relay with legacy fallback.
  BC-GPU-PANIC fixed — `Auto::new()` decoupled from test pool, returns `Err` instead
  of panicking. fault_injection SIGSEGV — `gpu-serial` override added to `stress`
  and `gpu` nextest profiles. Musl rebuild: fresh binaries with checksums.
  4,422 tests pass, all quality gates green.
- **Sprint 38: Deep Debt — BTSP Phase 2, Capability-Based Discovery & Idiom Sweep (Apr 9)**:
  BTSP Phase 2 connection authentication guard (`btsp` module) integrated into all
  accept loops. BearDog discovery evolved to capability-based (`discover_by_capability()`
  scans `*.json` discovery files for `btsp.session.create`). `Box<dyn Error>` →
  typed `BarracudaCoreError`. `#[allow]` → `#[expect]` with reason. `precision_brain.rs`
  smart-refactored (703→421L). 4 GPU test binaries serialized. Musl-static rebuild
  fixed (static-pie). Comprehensive audit: zero mocks in production, zero hardcoded
  primal names, all deps pure Rust. 4,421 tests pass, all quality gates green.
- **Sprint 37: Deep Debt — Test Module Refactor & Code Cleanup (Apr 8)**:
  `methods_tests.rs` (951L) smart-refactored into 6 domain-focused test modules.
  `buffer_test.rs` println! removed. `nadam_gpu.rs` stale comment removed.
  `force_interpolation.rs` indexed loop → iterator. 12-axis clean bill.
  4,207 tests pass, all quality gates green.
- **Sprint 36: Domain-Based Socket Naming & Flaky Test Serialization (Apr 8)**:
  Socket naming evolved from `barracuda.sock` to `math.sock` per
  `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3. Legacy symlink for backward compat. Domain
  field in `identity.get` evolved from `"compute"` to `"math"`. `three_springs_tests`
  serialized in gpu-serial nextest group (Mesa llvmpipe SIGSEGV mitigation).
  4,207 tests pass, all quality gates green.
- **Sprint 35: Deep Debt — Typed Errors, thiserror & Transport Refactor (Apr 8)**:
  `validate_insecure_guard` evolved from `Result<(), String>` to typed `BarracudaCoreError`.
  `PppmError` evolved to `#[derive(thiserror::Error)]`. `transport.rs` smart-refactored:
  380-line test module extracted to `transport_tests.rs` (866→490 LOC). 12-axis deep debt
  audit: clean bill on all axes. 4,207 tests pass, all quality gates green.
- **Sprint 34: BTSP Socket Naming & BIOMEOS_INSECURE Guard (Apr 8)**:
  Resolves GAP-MATRIX-12: `FAMILY_ID` socket scoping with standard env var precedence
  (`BARRACUDA_FAMILY_ID` → `FAMILY_ID` → `BIOMEOS_FAMILY_ID`), `BIOMEOS_SOCKET_DIR` support,
  `BIOMEOS_INSECURE` guard. Resolves GAP-MATRIX-06: plasmidBin metadata updated to v0.3.11.
  20 new BTSP compliance tests. 4,207 tests pass, all quality gates green.
- **Sprint 33: Wire Standard L2 Compliance (Apr 8)**:
  `capabilities.list` now returns Wire Standard L2 `{primal, version, methods}` envelope
  per `CAPABILITY_WIRE_STANDARD.md` v1.0, with `provided_capabilities` grouping derived
  from the dispatch table, `consumed_capabilities`, `protocol`, `transport`. New
  `identity.get` method returns `{primal, version, domain, license}` for biomeOS probes.
  Both JSON-RPC and tarpc paths wired. 31 methods (was 30). New `provided_capability_groups()`
  in discovery module derives structured capability groups with zero hardcoded domain catalog.
  13 new tests for L2 compliance. 4,187 tests pass, all quality gates green.
- **Sprint 32: Fault Injection SIGSEGV Resolution & Deep Debt Audit (Apr 7)**:
  Root-caused Mesa llvmpipe within-process thread safety SIGSEGV in 3 fault injection tests
  (concurrent GPU readbacks and unbounded OOM loop). Serialized GPU operations sequentially
  in `fault_concurrent_tensor_access`, `test_concurrent_error_handling`; bounded
  `fault_out_of_gpu_memory` to 256 iterations. Updated nextest coverage profile from
  deprecated `exclude = true` to `default-filter` syntax. Fixed 4 clippy findings
  (non-existent lint, protocol string, unfulfilled expects). Comprehensive 12-axis deep debt
  audit confirmed clean bill: zero production unsafe/unwrap/expect/println/mocks/hardcoding/
  TODO/`#[allow(`/commented-out code across all 1,116 Rust files. 4,180 tests pass (CI
  profile), 0 failures. All quality gates green.
- **Sprint 31: Deep Debt Cleanup & Test Stability Hardening (Apr 5)**:
  Removed deprecated `CoralReefDevice` alias (zero consumers). `SpirvError` evolved to
  thiserror derive. 12 dead_code reason strings corrected from "CPU reference path" to
  "public API — exercised by tests, available to downstream consumers". 11 additional
  SIGSEGV-prone test binaries gated behind `stress-tests` feature — `cargo test --workspace`
  now 100% clean. Comprehensive audit confirmed: zero production unwrap/expect/panic,
  zero hardcoded primal names, zero mocks in production, zero TODO/FIXME, all files under
  845 lines, all deps pure Rust. All gates green.
- **Sprint 30: Deep Debt Audit, Smart Refactoring & Test Stability (Apr 5)**:
  `executor.rs` smart-refactored (934→208L) into `executor.rs` + `invocation.rs` (756L);
  `DispatchCoords` struct eliminates `too_many_arguments`. SIGSEGV fix: `fhe_chaos_tests` +
  `fault_injection` excluded from coverage, `gpu-serial` test group serializes chaos/fault/property
  tests. Disabled `test_nn_vision_integration` evolved to `test_vision_pipeline_preprocessing`.
  Stale `NeuralNetwork` doc example cleaned. All gates green.
- **Sprint 29: Deep Debt Cleanup & Shader-First Evolution (Apr 4)**:
  Unified magic `256` workgroup size → `WORKGROUP_SIZE_1D` constant across 15+ files
  (shader dispatch, jackknife, biosignal, gradient, cpu_executor, all health ops, perlin,
  lattice ops). Removed dead `num-traits` from workspace deps. Smart refactoring:
  `executor.rs` 1,097→932 lines (vector ops extracted to `vector_ops.rs`), `eval_math`
  decomposed from 264-line monolith into 4 focused dispatch functions (eval.rs 629→527
  lines, `#[expect(too_many_lines)]` suppression eliminated). Production `expect()` in
  `wgpu_backend.rs` evolved to safe pattern-match + `Result` propagation. Misleading
  `nautilus/readout.rs` "no-op" documentation corrected to describe actual CPU ridge
  regression. `coralReef` documentation evolved to capability-based discovery language
  throughout coral_compiler module. Hardcoded `"biomeos"` and `"ecoPrimals"` namespace
  strings consolidated to shared constants across crates. Perlin noise 7× `#[expect]`
  blocks consolidated to 2 `perm_index` helper functions. All gates green (3,815 lib +
  16 naga-exec tests, 0 failures). Clippy pedantic+nursery clean, `cargo doc -D warnings`
  clean, `cargo fmt` clean.
- **Sprint 28: Zero-Copy ESN, Capability Naming & Error Evolution (Apr 4)**:
  5 `Tensor::clone().matmul()` → `matmul_ref()` zero-copy in ESN model. Runtime "coralReef" →
  "sovereign shader compiler" capability-based naming. tarpc `io::Error` source chain preserved via
  `From` impl. Comprehensive quality gate sweep: zero `#[allow(]`, zero `todo!()`, zero production
  unwrap, zero files >1000 lines, zero hardcoded sibling names in runtime. All gates green (4,446
  passing, 0 failures). Debris audit clean (no archive dir, no temp files, no stale scripts).
- **Sprint 27: primalSpring Audit Remediation & Doc Alignment (Apr 3)**:
  Hex bitwise literal fix (`0x3D`), `#[expect]` reason strings, barracuda-core lint promotions
  (`use_self`/`map_unwrap_or` → warn). Doc reconciliation: Rust files 1,116, tests 4,207+,
  SOVEREIGN_PIPELINE_TRACKER CPU interpreter updated from "Future" to "Shipped". All gates green.
- **Sprint 26: Comprehensive Audit, Refactor & Compliance (Apr 1)**:
  Full codebase audit against wateringHole standards. `executor.rs` smart refactor (1,020 → 886
  lines via WorkgroupMemory extraction to `workgroup.rs`). `cargo deny` bans fixed
  (`allow-wildcard-paths`). Stale `#[allow(clippy::module_name_repetitions)]` removed from
  naga-exec (never triggered). `#[allow(unused_async)]` → `#[expect(unused_async)]` in
  barracuda-core. Coverage measured at 80.54% line. Confirmed: zero production unwrap/panic/expect,
  all `.clone()` justified, discovery fully capability-based. Doc alignment across STATUS,
  CONVENTIONS, CONTRIBUTING, CHANGELOG. All quality gates green, 4,207+ tests, 0 failures.
- **Sprint 25: Deep Debt Evolution & Modern Idiomatic Rust (Mar 31)**:
  Zero production panics in naga-exec (5 `panic!` → `Result`). Zero production `.expect()` (6 sites
  → `Result`). All `#[allow(` → `#[expect(` with reason. barracuda-spirv `assert!` → `Result` + typed
  `SpirvError`. 5 index loops → idiomatic iterators. `submit_dispatch` → `submit_dispatch`
  (capability-based). `"biomeos"` → `ECOSYSTEM_SOCKET_DIR` constants. Showcase tokio `"1"` → `"1.50"`.
  `coral_compiler/mod.rs` 982 → 563 lines (smart test extraction). BC-01 Fitts variant param. BC-02
  Hick `include_no_choice`. BC-03 true 3D Perlin noise. BC-04 executor.rs 1,913 → 991 lines.
  All quality gates green, 4,100+ tests, 0 failures.
- **Sprint 24: WGSL-as-Truth + NagaExecutor + coralReef CPU Compilation (Mar 30)**:
  Major test architecture restructure: 337 GPU op test files migrated from `get_test_device_if_gpu_available()`
  to `get_test_device()`, enabling 2,770 tests to run on CPU/llvmpipe. 17 GPU-exclusive modules correctly
  re-gated. New crate `barracuda-naga-exec`: pure-Rust CPU interpreter for naga IR with f32/f64 native,
  workgroup shared memory, barriers, atomics (16 tests). `assert_shader_math!` and `assert_shader_math_f64!`
  macros for zero-GPU shader validation. coralReef IPC contract: 10 new wire types, 5 new `CoralCompiler`
  methods (`compile_cpu`, `execute_cpu`, `validate_shader`), capability discovery for `shader.compile.cpu`
  and `shader.validate`. `ShaderValidationBackend` enum with coralReef-first fallback chain. 4-layer
  validation architecture (llvmpipe / NagaExecutor / coralReef CPU / real GPU). 2,786 total tests,
  0 failures. All quality gates green.
- **Sprint 23: ludoSpring V35 Gap Resolution (Mar 29)**:
  15 new IPC methods wired (30 total). Socket path fixed to `barracuda.sock`.
  Dual-transport startup. All `#[allow(` migrated to `#[expect(`. 3,808 tests.
- **Sprint 22h: Deep Debt Evolution & Dependency Purge (Mar 29)**:
  Subgroup `subgroupAdd` reduction wired as top-tier path in `ReduceScalarPipeline`
  (3-tier: subgroup→DF64→scalar fallback). `enable f64;` directive removed from 47
  WGSL shaders (compile-time preamble injection handles it). `num-traits` external
  dependency eliminated — replaced with local `CpuFloat` trait in
  `shaders/precision/cpu.rs`. `LcgRng` consolidated from `spectral::anderson` to
  `crate::rng` (single source of truth for both anderson and lanczos modules).
  Hardcoded `"coralReef:"` log prefixes evolved to accurate `"naga"` labels.
  `const fn` promotions across rng module. 4,059 tests, 0 failures. Clippy clean.
- **Sprint 22g: Cross-Spring Deep Absorption (Mar 29)**:
  6 hotSpring WGSL shaders absorbed (TMU-accelerated Box-Muller PRNG, subgroup-accelerated
  reduce, ROP fixed-point force accumulation + conversion, shifted CG alpha + vector update).
  `tmu_tables.rs` TMU lookup table builder + `rop_force_accum.rs` fixed-point accumulation
  params. 3 healthSpring WGSL shaders absorbed (Hill dose-response, population PK Monte Carlo,
  Shannon/Simpson diversity) with full Rust dispatch modules + CPU reference implementations.
  `SiliconProfile` + `SiliconUnit` + `UnitThroughput` + `CompositionEntry` silicon personality
  model with tier-based workload routing. `has_subgroups` field + `wgpu::Features::SUBGROUP`
  feature negotiation for subgroup-accelerated shaders. 6-format capability parser absorbed
  from ludoSpring (Formats A–F for diverse primal response shapes) with semantic alias
  generation. Env-configurable IPC tolerances (`BARRACUDA_RPC_TIMEOUT_SECS`,
  `BARRACUDA_PROBE_TIMEOUT_MS`, `BARRACUDA_CONNECT_PROBE_TIMEOUT_MS`). 4,030+ tests,
  0 failures. Clippy pedantic+nursery clean.
- **Sprint 22f: PrecisionBrain-coralReef Integration (Mar 29)**:
  `PrecisionBrain` now accepts coralReef f64 lowering availability, routing
  F64/DF64 tiers as safe when coralReef sovereign compilation bypasses naga/NVVM.
  `needs_sovereign_compile()` method for callers. `CoralF64Capabilities` type +
  `capabilities_structured()` queries per-op polyfill availability from coralReef.
  `PrecisionAdvice` in `CompileWgslRequest` for informed coralReef decisions.
  Dispatch metadata (gpr_count, workgroup) wired into toadStool IPC. DF64
  sovereign routing: `compile_shader_df64` sends full source to coralReef before
  naga stripping when SPIR-V poisoning detected. 4,206 tests, 0 failures.
- **Sprint 22e: Probe Test Coverage & GPU Silicon Capability Matrix (Mar 29)**:
  14 new probe unit tests + 5 DeviceCapabilities tests. `GPU_SILICON_CAPABILITY_MATRIX.md`
  spec documenting FP64 rates (NVIDIA/AMD/Intel), DF64 strategy, toadStool VFIO
  silicon exposure. 4,194 tests.
- **Sprint 22d: f64 Transcendental Pipeline Awareness (Mar 29)**:
  Composite transcendental probes (NVVM JIT failure detection), `F64BuiltinCapabilities`
  evolved, `get_test_device_if_f64_transcendentals_available()` test gate, 10 failing
  tests → 0 via graceful skip. coralReef structured `CompileCapabilitiesResponse`.
- **Sprint 22c: IPC Evolution (Mar 29)**:
  Newline-delimited JSON-RPC framing (wateringHole v3.1). Unix socket discovery
  via `$XDG_RUNTIME_DIR/biomeos/shader.sock`. `biomeos` namespace alignment.
- **Sprint 22: Spring Absorption & Deep Debt Evolution (Mar 29)**:
  Critical fermion force sign fix (neg_eta convention) in 2 WGSL shaders.
  8 hotSpring WGSL shaders absorbed (5 multi-shift CG + 3 GPU-resident
  Hamiltonian/Metropolis/fermion-action). `gpu_multi_shift_cg.rs` orchestration
  with pipelines, buffers, and generic CPU reference. `gpu_resident_observables.rs`
  with O(1)-readback plaquette/KE/Hamiltonian/Metropolis pipeline. 6 RHMC/lattice
  tolerance constants. f32 Perlin 2D shader + API for ludoSpring. 32-bit LCG
  contract for ludoSpring. Lanczos eigenvector pipeline (`lanczos_eigenvectors`
  with Ritz vector construction) for groundSpring. All quality gates green
  (717 + 214 tests, clippy pedantic+nursery, doc).
- **Sprint 21: Compliance, Coverage & Validation-First Evolution (Mar 29)**:
  `health.liveness`, `health.readiness`, `capabilities.list` endpoints
  implemented per wateringHole Semantic Method Naming Standard v2.2.0
  (non-negotiable probes). All required aliases (`ping`, `health`, `status`,
  `check`, `capability.list`). `--port <PORT>` CLI flag per UniBin standard.
  Validation-first handler refactoring across JSON-RPC and tarpc — validate
  inputs before device check, enabling comprehensive testing without GPU.
  `barracuda-spirv` unsafe evolved to `#![deny(unsafe_code)]` + targeted
  `#[allow]`. Coverage 59.33% → 72.83% line (+13.5pp). 214 unit + 8 e2e
  tests in barracuda-core (up from 148). `rpc.rs` refactored (861→572
  lines via test extraction). `fhe_ntt_validation.rs` clippy cast_lossless
  resolved. All quality gates green.
- **Sprint 20: FMA Evolution & Lint Promotion (Mar 21)**:
  625 `suboptimal_flops` sites (415 lib + 210 test) evolved to `mul_add()` for hardware FMA
  precision. 4 clippy lints promoted from `allow` to `warn`: `suboptimal_flops` (415→0),
  `use_self` (332→0), `tuple_array_conversions` (2→0), `needless_range_loop` (45→0).
  All `needless_range_loop` sites evolved to idiomatic iterators (`.zip()`, `.enumerate()`,
  `.iter_mut()`). 232 files changed, 3,623+ tests pass, zero clippy errors.
- **Sprint 19: Deep Debt Solutions & Idiomatic Rust Evolution (Mar 21)**:
  Comprehensive audit across all dimensions: large files, mocks, hardcoding, unsafe, deps,
  TODO markers, `.unwrap()` in production. RPC `tolerances_get` evolved to centralized
  tolerance registry (by-name + by-tier lookup). Cast safety: all `usize as u32` in
  `TensorSession` replaced with `barracuda::cast::usize_as_u32()`. 6 new domain feature
  gates (`domain-fhe`, `domain-md`, `domain-lattice`, `domain-physics`, `domain-pharma`,
  `domain-genomics` for `ops::bio`). `FlatTree::validate()` evolved from `Result<(), &str>`
  to typed errors. Audit confirmed: zero unsafe, zero production `.unwrap()`/`.expect()`,
  zero TODO/FIXME markers, zero production mocks, all deps pure Rust, all large files
  justified (barrel modules or test-heavy). 3,623+ tests pass, zero clippy errors.
- **Sprint 18: Ecosystem Absorption & API Housekeeping (Mar 21)**:
  Full pull + review of 8 springs + 10+ primals. `GpuDriverProfile` struct removed (all
  springs migrated). `barracuda::cast` module with safe numeric casts and `CastOverflow`/
  `PrecisionLoss` typed errors. `ESN::wgpu_device()` and `MultiHeadEsn::wgpu_device()`
  accessors (neuralSpring request). `domain-fold` feature gate for structural biology
  shaders. f64 shader constants exposed as public API. Tolerance stability contract
  documented. `cast_lossless` lint promoted (zero violations). Ecosystem audit confirmed
  Hamming/Jaccard/L2, chi-squared/KL, xoshiro PRNG, HMM backward/Viterbi all already
  implemented. 3,618 tests pass, zero clippy warnings, all gates green.
- **Sprint 17: Nursery Linting, IPC Naming Evolution & Coverage Push (Mar 21)**:
  `clippy::nursery` blanket-enabled on both crates with 13 warnings fixed and
  scientific/GPU false positives selectively allowed in `Cargo.toml`. IPC method
  names evolved from `barracuda.{domain}.{operation}` to bare `{domain}.{operation}`
  per wateringHole Semantic Method Naming Standard — `REGISTERED_METHODS` constant
  (renamed from `METHOD_SUFFIXES`) + `normalize_method()` backward compatibility.
  13 pooling tests hardened: `test_pool::get_test_gpu_device()` with graceful skip
  on device-lost. Dead code audit: all 40+ `#[expect(dead_code)]` sites validated
  (CPU reference kernels, planned sovereign pipeline, Debug-derive usage). Coverage:
  71.59% line / 78.44% function / 69.37% region (up from 32%/59%). All gates green.
- **Sprint 15–16: Comprehensive Audit & Production Hardening (Mar 21)**:
  Device-lost detection evolution (`is_device_lost()` case-insensitive matching).
  Hardcoded domain lists eliminated in `primal.capabilities` (both JSON-RPC and tarpc
  now derive from `discovery::capabilities()` single source of truth). Lint evolution:
  42 `#[allow]` → 14 justified, `#[expect(reason)]` for all suppressions. Documentation
  accuracy: `discovery` module doc corrected from "mDNS scanning" to capability-based
  self-discovery. 20 new barracuda-core tests (lifecycle edge cases, error variant
  coverage, all 12 dispatch routes): 110 → 130 tests. Zero production `.unwrap()`
  confirmed across entire workspace (every `.unwrap()` is inside `#[cfg(test)]`).
  FHE test suite verified: 62 tests pass (prior failures were GPU contention).
  Hardware verification SIGSEGV resolved (GPU driver race under parallel execution).
  barracuda-core coverage: 68.73% function / 63.47% line. All quality gates green.
- **Sprint 14: Full vendor-agnostic evolution (Mar 21)**:
  `DeviceCapabilities` replaces `GpuDriverProfile` across 50+ files, `DeviceClass` replaces
  `GpuVendor`/`GpuDriver`, `SubstrateType::DiscreteGpu`/`IntegratedGpu` replaces vendor variants,
  `prefer_discrete()` replaces vendor preferences, +75 new tests (4,052+ total).
- **Deep debt sprint 14 — audit completion, doctest & hardware fixes (Mar 20)**:
  Fixed 2 pre-existing doctest failures (`complex_f64.rs` stale assertion,
  `sobol.rs` Rust 2024 edition `gen` keyword + merged doctest wrapper). Fixed
  hardware verification multi-GPU buffer lifetime panic (scoped tensors per
  device). 12 clippy new-edition lints fixed. SPDX header fix. Device-aware
  pooling test. 50 new tests (RBF surrogate, adaptive distance, Kimura, jackknife).
  108 doctests now all pass. 3,936 tests pass. All gates green.
- **Deep debt sprint 13 — comprehensive audit, coverage & test hardening (Mar 20)**:
  Full codebase audit against wateringHole standards with systematic execution.
  Cross-vendor GPU tolerance constants (`CROSS_VENDOR_MATMUL_F32_TOL`,
  `CROSS_VENDOR_ELEMENTWISE_F32_TOL`). FHE cold-start budgets
  (`NTT_N4096_COLD_BUDGET`, `FAST_POLY_MUL_N4096_COLD_BUDGET`). llvm-cov
  SIGSEGV resolved via nextest coverage profile excluding `hardware_verification`
  binary. 40+ new tests across `driver_profile`, `precision_brain`,
  `hardware_calibration`, `cubic_spline`, `solve`, `jackknife`. Stale
  `#[expect(clippy::unwrap_used)]` attributes removed. Coverage: 71.38% line /
  77.94% function (GPU-architectural ceiling on llvmpipe). Doc numbers aligned.
  3,886 tests pass. All gates green.
- **Deep debt sprint 12 — module decomposition & build optimisation (Mar 20)**:
  IPC `methods.rs` (675L) decomposed into `methods/` directory with 6 domain
  files. Hydrology `gpu.rs` (648L) decomposed into barrel + 3 pipeline files.
  Kernel router magic numbers evolved to named constants. `with_device_retry`
  double-permit fix restores full GPU test parallelism. Build profiles
  optimised (codegen-units=256, split-debuginfo, opt-level=2 for deps). 16 new
  tests (compute_graph + Lanczos). 3,555 tests pass. All gates green.
- **Deep debt sprint 7 — comprehensive audit & evolution (Mar 17)**: Smart
  module refactoring: `ode_bio/systems.rs` (744L) into `systems/` directory
  (5 per-system files, matching `params/` pattern). `gpu_hmc_trajectory.rs`
  (794L → 531L) with types extracted to `gpu_hmc_types.rs` (280L). 10
  `mul_add` evolutions in RK45 tolerance + cubic spline. 2 crate-level lint
  suppressions localized to per-site `#[expect(reason)]`. 28 new unit tests
  across 5 previously untested modules (utils, sparsity config/result,
  nn/config, session/types). Hardcoding evolution: transport defaults,
  discovery paths, resource quota presets → named constants. `cargo update`
  applied. `test_infinity_input` fixed with device-aware guard. 3,772 tests
  pass. All gates green.
- **Deep debt sprint 6 — cross-ecosystem absorption (Mar 16)**: GemmF64
  `execute_gemm_ex(trans_a, trans_b)` with WGSL `select()`-based stride swapping.
  FAMILY_ID socket paths per PRIMAL_IPC_PROTOCOL. blake3 `default-features=false`
  ecoBin pure. `deny.toml` `wildcards=deny`. `WGSL_MEAN_REDUCE` re-export for
  neuralSpring. 3 stale lint suppressions removed. 3,466 tests pass. All gates green.
- **Deep debt sprint 5 — audit execution & evolution (Mar 16)**: Comprehensive
  audit execution: 22 "for now" patterns evolved to proper engineering
  documentation with performance crossover thresholds. Device-lost panic handling
  DRY-refactored (4× duplicate eliminated). CUDA benchmark stub honestly
  relabeled as CPU baseline. Hardcoded bincount 256 → `DEFAULT_NUM_BINS`.
  21 new ODE bio system tests (all 5 models). Dependency analysis: all direct
  deps pure Rust, blake3 `pure` avoids C, rand 0.8/0.9 duplicate tracked.
  Zero-copy gap analysis documented (f64→f32 inherent, LSTM clone inherent).
  All quality gates green (fmt, clippy, doc, test compile).
- **Deep debt sprint 4 — sovereign wiring & zero-copy (Mar 15)**: SovereignDevice
  (then CoralReefDevice) wired to `compute.dispatch.submit` via capability-based discovery
  (`detect_dispatch_addr` scans `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
  `compute.dispatch`). Buffer staging via `BytesMut`. `dispatch_compute` evolved
  to `Entry` API. `Default` impl added. `# Errors` doc sections added. Pedantic
  lint promoted to `deny` in both crates. Tensor store `Mutex` → `RwLock`.
  Zero-copy: `CpuTensorStorage` → `BytesMut`, `EventCodec` → `Bytes`,
  `CompileResponse::into_bytes()`. Edition 2024 safety: `set_var` eliminated
  from tests. coralNAK → coralReef in all active docs. All quality gates green.
- **GPU streaming & comprehensive audit sprint (Mar 15)**: Full GPU submission
  pipeline refactored to eliminate blocking. `submit_and_poll_inner` split into
  separate `submit_commands_inner` + `poll_wait_inner` lock acquisitions —
  other threads interleave submits while one thread polls (eliminates 120s
  lock convoy on 16-thread nextest). 279 fire-and-forget GPU ops migrated from
  blocking `submit_and_poll` to non-blocking `submit_commands` (zero CPU wait
  for GPU-resident results). New `submit_and_map<T>` single-poll readback method
  collapses old double-poll into one `poll_safe` cycle. `read_buffer<T>` optimized
  internally. `--all-features` clippy compilation fixed (`is_coral_available`,
  `SovereignDevice::with_auto_device`, `has_dispatch`). Pre-existing lints fixed
  (doc_markdown in bcs/screened_coulomb/chi2, approx_constant in kinetics test,
  double_must_use in critical_screening/chi_squared). Full codebase audit: zero
  archive code, zero dead scripts, zero TODO/FIXME in production, zero files over
  1000 lines, zero .bak/.tmp debris. All quality gates green.
- **Deep debt sprint 3 — lint evolution & refactoring (Mar 14)**:
  `missing_errors_doc` and `missing_panics_doc` promoted to warn in both crates
  (zero violations). Cast lints (`cast_possible_truncation`, `cast_sign_loss`,
  `cast_precision_loss`, `cast_lossless`) promoted in barracuda-core.
  `large_stack_frames` documented as test framework artifact. `suboptimal_flops`
  evolved in all test files (mul_add with type annotations). `ode_bio/params.rs`
  refactored into 7-file modular structure. RBF `assemble_and_solve` zero-copy
  via `split_off`. CI: 80% coverage gate and chaos/fault tests now blocking;
  cross-compile job added for musl targets. Dead `ring` config removed from
  deny.toml. All quality gates green.
- **VFIO-primary architecture adoption (Mar 13)**: VFIO via toadStool adopted as
  primary GPU dispatch path. All root docs and specs updated. SovereignDevice
  (then CoralReefDevice) evolved to IPC-first architecture (no coral-gpu dependency). VFIO detection
  responsibility moved to toadStool (barraCuda queries via IPC). wgpu demoted to
  development/fallback.
- **Sovereign pipeline deep debt sprint (Mar 12)**: Hand-written `weighted_dot_df64.wgsl`
  (6 kernels with DF64 workgroup accumulators) replaces auto-rewrite for Hybrid devices.
  RHMC multi-shift CG + rational approximation + RHMC HMC absorbed from hotSpring into
  `ops::lattice::rhmc` and `ops::lattice::rhmc_hmc`. `@ilp_region` annotations added to
  high-value DF64 reduction shaders (variance_reduce_df64, weighted_dot_df64,
  mean_variance_df64, covariance_f64). Covariance f64 confirmed auto-rewrite safe
  (thread-local accumulators only). All quality gates green.
- **Deep debt sprint 2 — nursery lints & iterator evolution (Mar 12)**: 5 nursery
  lints promoted (redundant_clone, imprecise_flops, unnecessary_struct_initialization,
  derive_partial_eq_without_eq; suboptimal_flops kept allow with rationale). 193 files
  auto-fixed. All 7 if_same_then_else sites fixed and lint promoted to warn. Iterator
  evolution: csr diagonal, device_info NPU scan, fft_1d twiddle gen converted from
  range loops to idiomatic iterators. Discovery file paths derived from
  PRIMAL_NAMESPACE (3 sites). zeros/ones dispatch duplication eliminated via combined
  match arm. Total: 14 bulk-allowed lints now promoted (9 pedantic + 5 nursery).
  All quality gates green.
- **Comprehensive audit & deep debt sprint (Mar 12)**: Full codebase audit against
  wateringHole standards (uniBin, ecoBin, semantic naming, sovereignty, zero-copy,
  license compliance, code quality). 12-item remediation: `#![forbid(unsafe_code)]`
  in both crates; namespace-derived IPC method names via `PRIMAL_NAMESPACE` +
  `METHOD_SUFFIXES` (LazyLock); 648 WGSL SPDX headers added (806/806 complete);
  9 bulk-allowed pedantic lints promoted to warn (enforced); erfc_f64 recursion
  fix in stable_f64.wgsl; magic numbers extracted (CONSERVATIVE_GPR_COUNT,
  DEFAULT_WORKGROUP, CORAL_CACHE_ARCHITECTURES); zero-copy evolution
  (async_submit::read_bytes, ncbi_cache::load -> bytes::Bytes); unreachable! ->
  debug_assert! + graceful fallback; rustdoc zero warnings; BufferBinding import
  for --all-features clippy.
- **Sovereign dispatch wiring & deep debt evolution (Mar 11-12)**: Wired coral
  compiler cache → `SovereignDevice::dispatch_compute` (sovereign cache hits
  skip recompilation). Implemented `dispatch_binary` and `dispatch_kernel` on
  `SovereignDevice`. Added `PRIMAL_NAMESPACE` constant, replacing all hardcoded
  `"barracuda"` strings in IPC/socket/PID paths. Refactored `ode_generic` (890L →
  613L + 290L WGSL codegen). Cleaned 15 DF64 shader placeholder comments.
  Refactored CLI into modular subcommand handlers. Added `VoltaNoPmuFirmware`
  workaround detection. Eliminated double heap allocation in `Arc::from` across
  11 files. All clippy pedantic clean. External deps (pollster, futures, half)
  audited and justified. Zero production unwrap/expect confirmed.

Earlier completions (Mar 7–10) are documented in `CHANGELOG.md` and
`specs/REMAINING_WORK.md`.

---

## Immediate (P1)

- **PrecisionBrain → coralReef → SovereignDevice end-to-end test**: Integration test
  that exercises: brain routes to F64 via coral, `compile_wgsl_with_advice` sends
  `PrecisionAdvice`, sovereign dispatch sends gpr_count/workgroup metadata. Requires
  coralReef running in test CI.
- **DF64 NVK end-to-end verification**: Run df64 compilation on Yukawa force kernels through
  NVK/NAK on hardware. Validate sovereign compiler's safe WGSL roundtrip produces correct
  numerical results across all backends. Probe-aware `fp64_strategy()` is now in place to
  auto-fallback if native f64 fails.
- **Tensor core GEMM routing**: Eigensolvers/preconditioners via toadStool VFIO + coralReef
  HMMA/WGMMA emission for mixed-precision iterative refinement.
- **`BatchedTridiagEigh` GPU op**: groundSpring local QL implicit eigensolver is a candidate
  for absorption as a batched GPU tridiagonal eigenvector solver.
- **Multi-GPU OOM recovery**: `QuotaTracker` is wired into buffer allocation; next step
  is automatic workload migration when a device hits VRAM quota.
- **Kokkos parity validation baseline**: Document `sarkas_gpu` validation results, extract
  PPPM shader performance numbers for apples-to-apples comparison. Now unblocked by VFIO
  strategy — projected ~4,000 steps/s vs Kokkos 2,630 steps/s.

## Near-term (P2)

- **Test coverage to 90%**: Currently 80.54% line on llvmpipe (Sprint 26:
  measured via llvm-cov). CI 80% gate blocking (Sprint 3).
  Evolve `--fail-under` from 80 to 90 with real GPU hardware. Remaining gaps
  are exclusively GPU-dependent code paths.
- **Kokkos GPU parity benchmarks**: Run barraCuda GPU benchmarks on matching hardware,
  publish comparison data.

## Medium-term (P3)

- **Multi-GPU dispatch**: Evolve GpuView to span multiple devices with automatic work
  distribution across primary/secondary adapters.
- **Pipeline cache re-enable**: When wgpu provides a safe `create_pipeline_cache` API
  (or safe wrapper for `data: None`), re-enable in-memory pipeline caching. The field +
  accessor are preserved, `make_pipeline_cache` returns `None` until then.
- **Shader hot-reload**: File watcher for `.wgsl` files during development, automatic
  recompilation through sovereign pipeline.
- **Zero-copy evolution**: `bytes::Bytes` on I/O boundaries + `CpuTensorStorageSimple` +
  `CosineSimilarityF64` + RBF `assemble_and_solve` + `CpuTensorStorage` → `BytesMut` +
  `EventCodec` → `Bytes` + `CompileResponse::into_bytes()` done; remaining: pre-allocated
  buffers for `domain_ops.rs` CPU fallback clones, LSTM hidden state clones.

## Long-term (P4)

See `SOVEREIGN_PIPELINE_TRACKER.md` for the full sovereign pipeline tracker
including cross-primal dependencies, libc/musl → rustix evolution, and
cross-compilation target matrix.

- **Sovereign Compute Evolution**: Replace entire non-Rust GPU stack with coralReef
  as the unified compiler and driver for all GPU targets (eventually also the Rust
  compiler) via VFIO primary dispatch path (toadStool VFIO GPU backend + IOMMU isolation).
- **WebGPU browser target**: Compile barraCuda shaders for browser execution via wasm-pack
  and wgpu's WebGPU backend.
- **Distributed compute**: Cross-node GPU dispatch via primal-to-primal IPC for HPC clusters.

---

## C Dependency Chain — Evolution Map

**barraCuda has zero unsafe code and zero application-level C dependencies.**

The remaining C boundary is the OS/driver interface via transitive dependencies of
`wgpu` and `tokio`. These are system-level and do not constitute application C deps.

### barraCuda dependency chain (what touches C)

| Dependency | What it does | C boundary | Who evolves it |
|------------|-------------|------------|----------------|
| `wgpu` → `wgpu-hal` → `ash` → `libloading` | Vulkan FFI: dynamically loads `libvulkan.so` and calls the Vulkan C API | Vulkan driver (OS/GPU vendor) | **coralReef** (sovereign driver replaces Vulkan path) |
| `wgpu` → `wgpu-hal` → `renderdoc-sys` | RenderDoc debug capture FFI | Debug-only, never hits production | Can be feature-gated out of wgpu |
| `wgpu` → `wgpu-core` → `parking_lot_core` → `libc` | Futex/condvar syscalls for GPU synchronization | Kernel ABI, not a C library | Rust std evolves (already uses libc internally) |
| `tokio` → `mio` → `libc` | epoll/kqueue/io_uring syscalls | Kernel ABI | Rust std evolves |
| `tokio` → `signal-hook-registry` → `libc` | Signal handler registration | Kernel ABI | Rust std evolves |
| `getrandom` → `libc` | `/dev/urandom` or `getrandom(2)` syscall | Kernel ABI | Rust std evolves |
| `blake3` | Hashing (with `pure` feature) | **None** — `pure` flag = no C SIMD asm | Already pure Rust |

### coralReef dependency chain (what touches C)

| Dependency | What it does | C boundary | Who evolves it |
|------------|-------------|------------|----------------|
| `jsonrpsee` → `hyper` → `tokio` → `libc` | HTTP/WS transport + async runtime | Kernel ABI | Rust std evolves |
| `nak-ir-proc` (2 unsafe blocks) | `from_raw_parts` on `#[repr(C)]` struct fields with compile-time contiguity proofs | **None** — pure Rust, unsafe for performance | **coralReef** evolves: array-field pattern or `bytemuck` cast |

### The path to pure Rust end-to-end

Math is universal. A shader is just math. The execution substrate (GPU, CPU, NPU, Android
ARM core) is a hardware implementation detail — not a difference in universal math.

**Layer 1 — barraCuda (DONE)**: Zero unsafe, zero application C deps. WGSL shaders
express the math. The sovereign compiler optimises at the naga IR level in pure Rust.
Compilation flows through safe `create_shader_module`. The math layer is pure Rust today.

**Layer 2 — coralReef (2 unsafe blocks remain)**: The `nak-ir-proc` proc macro uses
`slice::from_raw_parts` on `#[repr(C)]` structs with compile-time contiguity proofs.
Evolution path: store matched fields as `[T; N]` arrays with named accessors, or use
`bytemuck::cast_ref`/`cast_mut` on Pod types. This is an internal coralReef evolution —
the IPC interface is unaffected.

**Layer 3 — GPU drivers (external, OS-level)**: `wgpu → ash → libvulkan.so` is the
system driver boundary. This is where the sovereign compute evolution eliminates the
last C dependency: coralReef's pure-Rust NVIDIA codegen replaces NAK, then
coralReef's driver layer replaces the Vulkan loader. The math never changes — only the substrate.

**Layer 4 — Kernel ABI (`libc`)**: Every Rust program on Linux calls the kernel through
`libc` (syscalls for memory, I/O, signals). This evolves via `rustix` (pure Rust syscalls
using `linux-raw-sys`) — see `SOVEREIGN_PIPELINE_TRACKER.md` for the phased evolution
from libc/musl to zero-package cross-compilation.
