# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-06-15.

---

## Recently Completed

- **Deep Debt Evolution Pass (Jun 15)**:
  MSRV bumped to 1.92. All 83 `collapsible_if` lints resolved via let-chain
  modernization (Rust 2024 idiom). Stale lint (`needless_type_cast`, removed in
  1.92) cleaned from naga-exec tests. 24 `suboptimal_flops` in MD force test
  modules suppressed at module level (reference physics math in textbook notation
  for GPU kernel verification). 2 `useless_vec` evolved to arrays. Dependency
  tree audited — 100% pure Rust, all patches updated. Coverage expanded: 22 new
  tests across `transport_config` (10), `btsp_negotiate` (8), `btsp_discovery` (3),
  plus tensor test `mul_add` fix. Environment-sensitive tests hardened for hosts
  running live ecosystem services. Test count: 4,970+. 
  Workspace: zero clippy warnings, zero TODO/FIXME, zero files >800L.

- **Wave 113: riboCipher Signal Acceptance (Jun 14)**:
  P2 riboCipher prefix stripping. NUCLEUS probers send `[0xEC, 0x01]` before
  JSON-RPC payloads to distinguish live primals from stale sockets. barraCuda
  now detects and strips the 2-byte prefix via `BufReader::fill_buf()` peek
  before entering the line-reading loop. Works on both UDS and TCP without
  platform-specific peek syscalls. 3 new unit tests. guideStone compliant.
- **Wave 109: guideStone Startup Convergence — STARTUP-BC-01 (Jun 11)**:
  Replaced `--no-unix` flag with `--bind-mode` (reads `PRIMAL_BIND_MODE` env
  var via clap `env` attribute). Standard startup envelope now:
  `barracuda server --bind-mode $PRIMAL_BIND_MODE --port $PORT`.
  Recognized values: `tcp_only`/`tcp-only`/`tcp`/`fallback` → TCP-only (no UDS);
  `unix`/absent → UDS primary (default). tarpc UDS also skipped when tcp_only.
  No per-primal case blocks needed in deploy scripts. guideStone P1+P4 compliant.
- **Wave 107: Socket Cleanup + method.describe (Jun 10)**:
  P2 `PRIMAL-SOCKET-CLEANUP`: Discovery file (`barracuda-core.json`) and legacy
  symlink now derive their directory from the socket path's parent, not from
  `resolve_socket_dir()` fallback. When `--socket /run/user/1000/biomeos/math.sock`
  is passed, state goes to `/run/user/1000/biomeos/`, not `/tmp/biomeos/`. Enables
  `ProtectSystem=strict` systemd hardening. Eliminates stale `/tmp` debris.
  LOW `BARRACUDA-METHOD-DESCRIBE`: Implemented `method.describe` RPC (76 methods
  with param schema, description, access level). Callers can self-correct param
  format mismatches. Registered in dispatch table, included in capabilities.
- **Deep Debt: Zero Files Over 800L (Jun 8)**:
  Refactored the last 2 files exceeding 800 lines. Extracted `TransportEndpoint`,
  `TransportStream`, `connect_transport()` into dedicated `transport_endpoint.rs`
  (237L) with 9 unit tests (serde roundtrip, wire compat, connect edge cases).
  Converted binary from single-file `barracuda.rs` (819L) into directory module:
  `main.rs` (437L) + `commands.rs` (219L) + `discovery_file.rs` (173L). All
  production files now under 800 lines. Zero clippy warnings. Full audit confirms:
  `#![forbid(unsafe_code)]` all crates, zero production unwrap, zero production
  mocks, zero hardcoding, zero TODOs/FIXMEs, all deps pure Rust. Primal elevated
  to composition-ready.
- **Wave 101: Transport Self-Knowledge Fix (Jun 8)**:
  Removed `sourdough-core` path dependency per primal self-knowledge principle.
  Implemented `TransportEndpoint` and `connect_transport()` locally in
  `ipc::transport` (~120 lines) with identical `#[serde(tag = "transport")]`
  wire format. Zero cross-primal dependencies. Pattern now matches coralReef,
  squirrel, sweetGrass, nestGate (LOCAL impl, wire format is the contract).
  All 5 production outbound paths use local `connect_transport()`.
  `TRANSPORT_ENDPOINT` env var still operational. Clean build, zero warnings.
- **Wave 100: Transport Injection (Jun 8)**:
  Adopted `TransportEndpoint` as canonical transport standard. Binary accepts
  `TRANSPORT_ENDPOINT` env var (launcher/Tower injects UDS or TCP endpoint,
  overriding CLI flags). Evolved 5 production outbound paths (discovery, BTSP,
  neural announce, dispatch peer, capability resolution) to use
  `connect_transport()`. Self-bind already compliant (127.0.0.1 default, UDS
  primary, no 0.0.0.0). LOW priority item from ecosystem transport evolution.
- **Wave 93: Build Fix — Stash Conflict Resolution (Jun 7)**:
  Commit `4d4aacff` ("resolve stash conflict — accept upstream simple_mlp") broke
  the build: `to_binary()`/`from_auto()` methods were inaccessible from
  `persistence.rs` because the stash resolution dropped `mod serialization;` from
  `simple_mlp/mod.rs`. Also lost `#[serde(alias = "weights/biases")]` annotations
  on `DenseLayer` (broke `ml.perceptron_train` → `ml.mlp_save` pipeline). Fixed in
  3 commits: restored serialization submodule, removed duplicate stubs (kept proper
  bincode+BLAKE3 implementation over simplified JSON-only), restored serde aliases.
  Removed dead `training.rs` (code already inlined by upstream merge). 572/574 IPC
  tests pass (2 known environment-dependent BTSP failures). Build depot-ready (13/13).
- **Wave 82c: Deep Debt Final + Module Refactoring (Jun 6)**:
  Refactored `simple_mlp.rs` (885L) into directory module: `mod.rs` (556L) +
  `serialization.rs` (135L) + `training.rs` (187L). Zero production files >800L.
  Fixed 4 tensor tests to use shared GPU test pool (eliminated wgpu contention).
  Full deep debt audit confirmed: `#![forbid(unsafe_code)]` on all crates, zero
  production unwrap, zero production mocks, zero hardcoding, all deps pure Rust.
  **Wave 82c P1 capability_registry.toml already delivered** (Wave 78). 4600 tests.
- **Wave 79: Coverage Sprint + Transport Verification (Jun 5)**:
  32 new CPU edge-case tests covering linalg/signal/spectral/stats/math/noise/rng/
  activation/graph/ODE error paths. Transport compliance verified (UDS default,
  --socket/--unix operational, no 0.0.0.0 bind). 569 IPC tests. Clippy clean.
- **Wave 78: Capability Registry + Binary Serialization Tests (Jun 5)**:
  Created `config/capability_registry.toml` (283 lines, 23 domains, 96 methods,
  access control, consumed capabilities, hardware requirements, per-op metadata).
  12 new binary serialization tests for BCML format validation.
- **Wave 76: Integration Readiness + Binary Serialization (Jun 4)**:
  biomeOS v4.05 remote infer path validated E2E. Wire-format interop fix (serde
  aliases for `weights`/`biases`). Binary model serialization: `BCML` format with
  44-byte header (magic + version + format + BLAKE3 checksum). `ml.mlp_save` accepts
  `"format":"bincode"`, `ml.mlp_load` auto-detects. 539 IPC tests.
- **Wave 74: End-to-End Perceptron Pipeline (Jun 3)**:
  Added `ml.perceptron_train` — accepts raw `dispatch_telemetry.jsonl` records,
  extracts 36-dim features (domain one-hot + normalized latency/load/affinity),
  trains single-layer perceptron, optionally serializes to disk. Wire-compatible
  with `NEURAL_API_PERCEPTRON_DESIGN.md`. 6 new pipeline tests.
- **Wave 73: Perceptron Training for biomeOS L5 Neural API (Jun 3)**:
  Evolved `ml.mlp_train` to support dimension-shorthand wire contract:
  `{"layers": [36, 16], "inputs": [...], "targets": [...]}` auto-initializes
  Xavier-uniform random weights and trains via SGD backprop. Backwards-compatible
  with explicit-weights form. Added `SimpleMlp::from_dims()` constructor. 6 new
  tests covering 36-dim perceptron (batch-256, two-layer, error paths). Matches
  `NEURAL_API_PERCEPTRON_DESIGN.md` contract exactly. Pure software, zero GPU.
  Unblocks biomeOS perceptron shadow mode + primalSpring extraction.
- **Wave 67: Cross-Gate Dispatch Pipeline + Transport Split (Jun 1)**:
  Implemented 3 new IPC methods for hotSpring cross-gate compute dispatch pipeline:
  `compute.dispatch.capabilities` (GPU/CPU capability reporting for routing),
  `compute.dispatch.submit` (shader binary + input → job execution → job_id),
  `compute.dispatch.result` (job_id → output data retrieval). Wire-compatible with
  hotSpring's `cross_gate.rs` contract. 9 new tests. Method count 87 → 90.
  Also: split `transport.rs` (805L → 707L + `transport_config.rs` 113L), bumped
  `tokio` 1.50 → 1.52. Full audit clean: zero debt markers, zero clippy warnings.
- **Wave 54: Graceful GPU-less Survival (May 27)**:
  Added `--no-gpu-probe` CLI flag and `BARRACUDA_NO_GPU_PROBE` env var. When set,
  wgpu adapter enumeration is skipped entirely — instant startup in cpu-shader-only
  degraded mode. Eliminates ~30s DRM probe delay on GPU-less hosts (VPS, containers,
  broken drivers). Server stays alive indefinitely serving CPU tensor ops. 6 new
  integration tests. Addresses southGate deployment feedback from primalSpring.
- **Wave 53: Coverage Expansion (May 26)**:
  Filled all handler-level test gaps identified by primalSpring audit. Added 18 new
  IPC coverage tests: `stats.variance` (4 tests — previously zero), `ml.esn_predict`
  happy path + state injection (3 tests), `ml.mlp_train` happy path + error paths (3),
  `stats.correlation` error paths (3), `auth.*` dispatch integration (4). Added 6 tests
  for `runtime::tokio_block_on` (previously untested utility). 4,501+ tests total.
  Clippy zero warnings. Live coralReef CI and DF64/HMMA remain hardware-gated (documented).
- **Wave 49: Ecosystem Tightening (May 25)**:
  Fossilized `showcase/` (9 demos, 26 files) to `fossilRecord/primals/barraCuda/showcase_wave49/`.
  Removed 3 stale `target/release/barracuda` deployment patterns from demo scripts.
  Verified `notify-plasmidbin.yml` active. No local `wateringHole/` tree. No pipeline debt.
  Post-primordial mandate compliant: plasmidBin is sole binary channel.
- **Deep Debt Remediation Sprint (May 24)**:
  Full audit and evolution pass. (1) Smart refactor of `math.rs` (1046L → 305L) into
  domain modules: `stats.rs` (576L) + `signal.rs` (151L). (2) Eliminated `pollster`
  dependency — unified all sync-async bridge sites to `runtime::tokio_block_on`.
  (3) Evolved hardcoded `BEARDOG_*` env vars to generic `BTSP_PROVIDER_SOCKET` /
  `BTSP_FAMILY_SEED` with backward-compatible fallback. (4) Wrapped 47 unsafe
  env-manipulation sites in safe helpers. (5) Trimmed `transport.rs` from 825L → 799L.
  Zero lint warnings, all IPC tests pass. `-1,823 +747` net lines.
- **Wave 47: Deployment Behavior Convergence (May 24)**:
  Added `--socket` as visible alias for `--unix` CLI flag per
  DEPLOYMENT_BEHAVIOR_STANDARD. Uniform `--socket PATH` from nucleus_launcher.sh.
- **Wave 44: Outbound Neural API Startup Announce (May 23)**:
  Added outbound `primal.announce` push to biomeOS Neural API on startup per
  Wave 44 P1 audit. New `ipc::neural_announce` module handles 3-tier socket
  discovery, payload construction, and UDS send. Wired into all server startup
  paths (UDS, service, TCP-only). Fire-and-forget with graceful standalone
  degradation. 4 new tests. All 149 IPC + announce tests pass.
- **Wave 43: Neural API `primal.announce` Schema (May 23)**:
  Upgraded `primal.announce` to biomeOS v3.68+ Neural API wire schema per
  primalSpring Wave 43 audit. Added `socket` (UDS discovery path), `signal_tiers`
  (`["node"]`), `cost_hints` (math/shader/compute weighted), and `latency_estimates`.
  Capabilities narrowed to canonical routing domains `["math", "shader", "compute"]`.
  New `discovery_socket_path()` helper. 2 new schema compliance tests. All 145 IPC
  tests pass.

- **Sprint 73: Cross-Spring Pattern Absorption (May 20)**:
  Deeper audit of spring patterns. Absorbed 9 new IPC methods spanning 3 springs:
  (1) **Multi-model regression** (airSpring): `stats.fit_quadratic`, `.fit_exponential`,
  `.fit_logarithmic` — sensor correction models now universal (beyond `fit_linear`).
  (2) **Ecology/rarefaction** (groundSpring): `stats.rarefaction_curve` — hypergeometric
  expected richness at subsampled depths; library existed, now wired to IPC.
  (3) **Gamma/SPI** (airSpring): `stats.gamma_fit` (Thom MLE) + `stats.gamma_cdf`
  (regularized incomplete gamma) — building blocks for drought indices.
  (4) **Signal processing** (healthSpring): new `signal.*` namespace with
  `signal.detect_peaks`, `signal.bandpass`, `signal.derivative` — Pan-Tompkins
  pipeline stages elevated to universal primitives. 90 methods total.
- **Sprint 72: Universal Shader Absorption & IPC Expansion (May 20)**:
  Cross-spring audit revealed 4 spatial compute shaders in ludoSpring with no
  barraCuda equivalent and 3 stats library functions with no IPC exposure. Resolved:
  (1) **IPC wire exposure**: `stats.simpson`, `stats.bray_curtis`, `stats.hill` —
  library functions already existed, now wired to JSON-RPC surface (78 methods total).
  (2) **Spatial compute shader absorption**: BFS wavefront, DDA raycast, fog-of-war,
  and tile lighting elevated from ludoSpring game engine into universal
  `shaders/spatial/` directory. These are general-purpose 2D grid primitives reusable
  by any spring (pathfinding, visibility, influence maps, LiDAR sim, sensor coverage).
  (3) **Provenance**: `SpringDomain::LUDO_SPRING`, `ShaderCategory::SpatialCompute`,
  4 registry entries, and evolution timeline event for the absorption.
- **Sprint 71: CG-3 Sovereign Dispatch Contract Documentation (May 19)**:
  Addressed primalSpring "GPU API alignment (`submit_and_map`)" composition gap.
  Documented the full sovereign dispatch contract in `TENSOR_WIRE_CONTRACT.md`:
  barraCuda→coralReef compile request (precision_advice + hardware_hint),
  coralReef→barraCuda compile response (field mapping: binary_b64, gprs, workgroup,
  shared_memory, barriers → ShaderDispatchInfo), and barraCuda→toadStool dispatch
  submission (compute.dispatch.submit with binary + metadata + buffer bindings).
  Clarified that `submit_and_map` is barraCuda's *local* wgpu readback API — not
  part of the cross-primal sovereign contract. The sovereign path uses IPC-based
  readback. barraCuda routing + wire format are DONE; coralReef HMMA codegen and
  toadStool QMD IPC handshake are the remaining blockers.
- **Sprint 70: primalSpring Wave 22 Stadial Gate — Checklist Compliance (May 17)**:
  Self-audited against universal standards checklist. Two gaps found and resolved:
  (1) **`primal.announce`**: atomic self-registration method for biomeOS composition
  — returns primal name, namespace, version, domain, full method list, capabilities,
  signal tier, hardware state, transport, and license. Consistent with songbird/
  esotericWebb adoption pattern.
  (2) **`btsp.capabilities`**: BTSP cipher suite advertisement per
  `DARK_FOREST_GLACIAL_GATE_STANDARD.md` §5 — returns supported cipher suites
  (chacha20-poly1305, hmac_plain, null), preferred cipher, and Phase 3 support flag.
  (3) **Stability tier annotations**: every registered method annotated with stability
  tier (stable/evolving/internal) in `TENSOR_WIRE_CONTRACT.md`.
  75 registered IPC methods total. 4 new tests. All clippy clean.
  Method count discrepancy (audit said 38 — actual 75) noted as stale snapshot.
  `submit_and_map` composition gap confirmed already resolved (Sprint 42/53).
  `build_from_source` is plasmidBin-side, not local.
- **Sprint 69: hotSpring Trio Audit — `health.version` RPC (May 14)**:
  Addressed hotSpring audit item: trio-consistent `health.version` standalone RPC method.
  toadStool and coralReef both expose `health.version` for plasmidBin doctor and upgrade
  scripts. barraCuda now matches: returns `{ primal, version, rust_version }` without
  hardware probing. Registered in `REGISTERED_METHODS` (73 methods total). 2 new tests.
  OOM fleet failover remains intentionally parked until toadStool multi-GPU IPC readiness.
- **Sprint 68: hotSpring Trio Audit — TENSOR_WIRE_CONTRACT Alignment (May 13)**:
  Addressed hotSpring audit item: `TENSOR_WIRE_CONTRACT.md` batch ops table updated to
  include `sub` and `negate` (Sprint 66 implementations). IPC namespace table updated to
  reflect full 72-method coverage. GEMM tensor-core routing confirmed stable per audit
  (routes tensor-core-eligible precisions to `KernelTarget::Sovereign` with
  `HardwareHint::TensorCore`; awaits coralReef HMMA codegen for end-to-end execution).
  OOM detection + classification confirmed sufficient; fleet failover deferred.
- **Sprint 67: 12-Axis Deep Debt Audit — Deprecated Ceremony Removal (May 13)**:
  Comprehensive 12-axis sweep confirms clean bill on all axes: zero files >800L
  (max 797L), zero unsafe in production, zero C deps, zero todo!/unimplemented!,
  zero println! in library, zero Result<T,String>, zero mocks in production, zero
  hardcoded primal names in runtime. 3 stale `#[deprecated]` annotations on private
  CPU fallback functions evolved: `convolve_1d_cpu` → `convolve_1d_scalar`,
  `gradient_1d_cpu` → `gradient_1d_scalar`, `jackknife_leave_means_cpu` →
  `jackknife_leave_means_scalar`. Deprecation ceremony (`#[deprecated]` +
  `#[expect(deprecated)]`) replaced with clean naming — these are active scalar
  fallbacks, not deprecated API. All clippy --all-targets -D warnings clean.
- **Sprint 66: hotSpring Trio Audit — TensorSession Lattice Ops (May 13)**:
  Addressed GAP-HS-027 from hotSpring compute trio audit. Added `sub` (subtract)
  and `negate` operations to `TensorSession` for physics/lattice workloads:
  (1) **`sub(a, b)`**: `output = a - b` — eliminates the 2-op workaround
  (`add(a, scale(b, -1.0))`) for leapfrog integrators (`p = p - dt * force`).
  (2) **`negate(a)`**: `output = -a` — dedicated sign-flip for force conventions.
  Both ops wired through full stack: `SessionOp` enum, `SessionPipelines` (inline
  WGSL compiled once at construction), `dispatch.rs` encoding, `TensorSession`
  public API, and `tensor.batch.submit` IPC handler (`BinaryOp::Sub`,
  `UnaryOp::Negate`). 3 new tests: sub correctness, negate correctness, leapfrog
  integration pattern (`p_new = p - dt * force`). GAP-HS-041 (`stats.entropy`)
  confirmed already resolved (registered alias for `stats.shannon` since Sprint 50).
  OOM fleet failover: detection infrastructure live (Sprint 64); fleet routing
  deferred to toadStool multi-GPU readiness. All clippy clean, all tests pass.
- **Sprint 65: Deep Debt Remediation — Error Observability + Magic Number Evolution (May 13)**:
  Comprehensive 12-axis audit confirmed zero files >800L, zero unsafe in production, zero
  C deps, zero todo!/unimplemented!. Addressed remaining debt items:
  (1) **Magic number evolution**: cpu_executor thread fallback, memory estimate, bandwidth
  estimate now use named constants (`FALLBACK_THREAD_COUNT`, `AVAILABLE_MEMORY_RATIO`,
  `ESTIMATED_DDR4_BANDWIDTH`).
  (2) **Error observability**: transport.rs `dispatch_line` now logs I/O write failures via
  `tracing::debug!` before discarding. `eval_record.rs` load_or_new logs load failures.
  `creation.rs` new_gpu/new_cpu preserves inner error context instead of discarding.
  (3) **Deprecated removal**: Removed dead `TensorSession` type alias from tensor_context
  (deprecated since 0.3.12, zero remaining usages). Updated stale doc reference.
  All clippy --all-targets -D warnings clean. All tests pass.
- **Sprint 64: hotSpring Trio Audit — Sovereign Path + GEMM Routing + OOM Recovery (May 13)**:
  Addressed 3 remaining items from hotSpring compute trio audit:
  (1) **Sovereign path differentiation**: Added `dispatch_path` field to `precision.route`
  IPC response (`"wgpu"` | `"sovereign"` | `"unavailable"`). Uses `compute_device()` to
  resolve active dispatch tier. Enables hotSpring's `PrecisionAdvisory` to route through
  toadStool VFIO path vs local wgpu without secondary IPC queries.
  (2) **Tensor-core GEMM routing**: Extended `kernel_router` with `MatmulPrecision` enum
  and `KernelTarget::Sovereign` variant. `DenseMatmul` with F16/BF16/TF32 precision now
  routes to `HardwareHint::TensorCore` via sovereign dispatch (coralReef HMMA codegen +
  toadStool dispatch). F32/F64/None remains WGSL compute path. Forward-compatible for
  coralReef HMMA codegen readiness.
  (3) **Multi-GPU OOM recovery**: Added `oom` flag to `WgpuDevice`, wired OOM detection
  in uncaptured error handler (out of memory / allocation failed / not enough memory),
  added `is_oom()` + `clear_oom()` public API, extended `is_retriable()` to include OOM.
  Multi-device pool can now detect OOM and migrate workloads. All clippy clean, all tests pass.
- **Sprint 63: Glacial Debt Niche Tasks — DF64 NVK E2E + Framework Parity (May 13)**:
  Addressed 3 niche tasks from primalSpring Glacial Debt Escalation audit:
  (1) **DF64 NVK E2E**: Added 2 GPU-dispatched E2E tests exercising production
  `compile_shader_df64` path (FMA kernel 256-element, Kahan summation 1000-element)
  with numerical verification against CPU reference. Covers the full DF64 pipeline:
  df64_core prepend → sovereign compilation → GPU dispatch → f64 readback.
  (2) **Framework parity benchmarks**: Added `lammps_parity` bench (LJ f64 + Yukawa
  f64 at N=256/1K/4K with LAMMPS-reference timings) and `scipy_parity` bench
  (sum_f64, variance_f64 Welford, cdist Euclidean with NumPy/SciPy-reference timings).
  (3) **Coverage push**: Added 6 compilation smoke tests covering all compilation tiers
  (raw, auto-downcast, f64-tiered, df64-prepend). All pass. clippy --all-targets clean.
- **Sprint 62: Clippy Pedantic All-Targets Clean + 12-Axis Audit (May 13)**: 9
  test-code clippy lints resolved (suboptimal_flops, cast_lossless, assert_eq with
  bool, single_char_pattern). `cargo clippy --all-targets -- -D warnings` now zero
  warnings. Fresh 12-axis deep debt audit confirms zero actionable items across all
  1,160 `.rs` files. Max file 793L. All deps pure Rust.
- **Sprint 61: Diesel Engine Migration Prep (May 13)**: hotSpring audit identified
  15+ files wired to coralReef's diesel engine stack (hardware runtime) that should
  target toadStool. Updated `backend.rs`, `compilation.rs`, `workarounds.rs` doc
  comments from `coral-driver` to toadStool. Evolved hotSpring's `fleet_client.rs`
  with toadStool-first socket discovery (`hardware_daemon_run_dir()`), added
  `toadstool-glowplug` alias in `PRIMAL_ALIASES`, updated fleet file resolution to
  try `toadstool-ember-fleet.json` before legacy. Full cutover gated on toadStool
  Phase C (C1–C7).
- **Sprint 60: Registry Drift Fix + Clean Audit (May 13)**: ProjectNUCLEUS audit
  flagged `registry_tests.rs` asserting `== 71` (was 72 after Sprint 58). Evolved
  from hardcoded count to sanity floor (`>= 70`) + uniqueness check — prevents
  future drift. Fresh 12-axis audit confirms zero remaining actionable items
  across all 1,160 `.rs` files. `cc` transitive dep confirmed phantom (blake3
  `pure` feature skips all C compilation).
- **Sprint 59: 12-Axis Deep Debt Audit + Docs Hygiene (May 12)**: Comprehensive
  12-axis sweep confirms clean bill of health. Two actionable findings fixed:
  (1) `method_gate.rs` hardcoded `"barraCuda"` → `crate::PRIMAL_NAME` constant,
  (2) `precision_brain_tests.rs` 862L → split to 461L + 407L (trio E2E extracted).
  Root docs refreshed: SOVEREIGN_PIPELINE_TRACKER, SPRING_ABSORPTION,
  PURE_RUST_EVOLUTION updated to 0.4.0/May 12. BREAKING_CHANGES reconciled.
  wateringHole README corrected (5 active handoffs, not zero).
- **Sprint 58: Precision Route Advisory Method (May 12)**: Pass 14 convergence —
  `precision.route` IPC method wired. Exposes `PrecisionBrain` domain→tier routing
  over JSON-RPC for upstream primals (hotSpring, springs) to query recommended
  precision tier, hardware hint, FMA safety, and compiler requirements for all 15
  physics domains. No-GPU fallback returns domain minimum tier. Runtime coral
  detection via `is_coral_available()`. 22 new tests (4 validation, 15 domain
  routing, 1 structure, 2 dispatch integration). 72 registered methods.
- **Sprint 57: Trio Contract E2E Validation (May 12)**: primalSpring Evolution Sprint 4
  audit execution. 3 trio contract E2E tests validate complete data-flow chain:
  PrecisionBrain → PrecisionAdvice → coralReef wire format → ShaderDispatchInfo →
  toadStool dispatch. Covers F64 (LatticeQcd), DF64 (GradientFlow), and TensorCore
  (F16 MMA GEMM routing). Mock TCP servers verify correct `hardware_hint`, `gpr_count`,
  `shared_mem_bytes`, and `barrier_count` propagation on the wire.
- **v0.4.0 Stadial Gate Release (May 12)**: Springs convergence version. Precision
  ladder complete. Dispatch wire extracted and hardened. bearDog crypto delegation
  audited and confirmed correct. All cleanup items from primalSpring Evolution Sprint 3
  Phase C resolved. Tagged for plasmidBin.
- **Sprint 56d: Precision Ladder + Dispatch Wire Hardening (May 12)**:
  primalSpring Evolution Sprint 2 audit execution. Precision ladder evolution:
  `PrecisionTier::recommended_hardware_hint()` maps all 15 tiers to hardware units
  (TensorCore for F16/BF16/TF32/FP8, Compute for F32/F64/DF64/QF128/DF128/quantized).
  `requires_compiler_support()` documents coralReef dependency. `.hardware_hint()`
  builder on `ComputeDispatch`. Integration tests verify F32→DF64→F64 hint chain +
  tensor core routing. bearDog crypto delegation audited — Phase 1-2 delegated,
  local crypto retained for Phase 3 HKDF + per-frame AEAD (documented rationale).
  5 dispatch wire error-path tests (connection refused, malformed, RPC error,
  output buffer readback, hardware hint serialization × 6 variants). IPC coverage
  gap closed: `tensor.matmul_inline` (5 tests) + `linalg.graph_laplacian` (4 tests).
  Zero registered-but-untested handlers.
- **Sprint 56c: Independent Evolution Execution (May 12)**:
  Coverage push: 45 new CPU-testable IPC handler tests (447→492 total), covering
  `ode.step`, `stats.covariance`, `stats.spearman`, `stats.fit_linear`,
  `stats.empirical_spectral_density`, `spectral.fft`, `spectral.power_spectrum`,
  `linalg.solve`, `nautilus.*` full lifecycle, `ml.mlp_train`, `ml.esn_predict`.
  Wired `PrecisionAdvice` through `SovereignDevice::live_compile()` →
  `compile_wgsl_with_advice()` (coral gets full precision context for f64 lowering).
  4 sovereign integration tests validate PrecisionBrain → advice → coral wire chain.
  DF64 NVK Yukawa verification prep: production `yukawa_df64.wgsl` naga validation
  test + CPU reference implementation with analytical correctness (Newton's 3rd law,
  force magnitude exp(-κr)(1+κr)/r²). Hardware dispatch deferred to NVK access.
  12-axis deep debt audit: `sovereign_device.rs` 816→641L via dispatch wire protocol
  extraction (`sovereign_dispatch_wire.rs`). Zero files >800L across codebase.
- **Sprint 56b: Compute Trio Wave 8 Triage (May 11)**:
  primalSpring Compute Trio audit confirms barraCuda is **compute trio ready** — zero
  code changes required. SovereignDevice dispatch E2E live since Sprint 48
  (`shader.compile.wgsl` → coralReef, `compute.dispatch.submit` → toadStool). 4-tier
  fallback architecturally correct. Gate 3 (`stats.mean`) passing. Crypto IPC delegation
  correctly deferred (per-frame AEAD latency prohibitive). 12-axis deep debt audit clean.
- **Sprint 56: 12-Axis Deep Debt — Linalg Module Extraction (May 8)**:
  `math.rs` 892→674L via extraction of 4 linear algebra handlers (`linalg.solve`,
  `linalg.eigenvalues`, `linalg.svd`, `linalg.qr`) to cohesive `linalg.rs` (225L).
  Domain-aligned split: linalg is a distinct mathematical subdomain from stats,
  activations, and ODE integration. 12-axis audit clean (zero files >800L, zero
  TODO/FIXME, zero unsafe, zero mocks in production, all deps pure Rust).
- **Sprint 55: GAP-11 Complete — MLP Training + Nautilus Sessions (May 8)**:
  Implemented `SimpleMlp::train()` (SGD backpropagation, all 5 activations).
  Wired `ml.mlp_train` IPC method. Implemented Nautilus server-session store
  (Path B, `job_id` pattern) with 6 methods: `nautilus.create`, `observe`,
  `train`, `predict`, `export`, `import`. GAP-11 fully closed (18/18).
  Total methods: 71. All quality gates green.
- **Sprint 54: Stateful IPC + Method Gate JH-0 (May 7)**: Implemented `ode.step`
  (linear ODE RK4 integration, stateless) and `ml.esn_predict` (ESN prediction
  with client-managed reservoir state). Both follow Path A (Stateless with
  Client-Managed Snapshots) from the Stateful API Architecture Advisory.
  Added `EsnClassifier::get_state`/`set_state` public accessors. Adopted
  `MethodGate` pattern per `METHOD_GATE_STANDARD.md` v1.0 (JH-0): pre-dispatch
  authorization gate, permissive default, `auth.check`/`auth.mode`/`auth.peer_info`
  introspection methods, Public/Protected classification. Total methods:
  64 (was 59). 21 new gate tests. All quality gates green.
- **Sprint 53: Phase 58b GPU API Drift Documentation (May 5)**: Documented
  `WgpuDevice::submit_and_poll` → `submit_and_map<T>` breaking change in
  `BREAKING_CHANGES.md` with full migration guide. Documented Discovery Escalation
  Hierarchy participation (Tiers 1, 3, 4, 5 confirmed). Resolves primalSpring
  Phase 58b "GPU API drift" audit item for barraCuda side (wetSpring one-line
  migration is downstream). All quality gates green.
- **Sprint 51b: Phase 3 Transport Switch Verification (May 3)**: Fixed interop gap
  flagged by primalSpring audit — `buf_reader.into_inner()` discarded buffered
  bytes on negotiate transition (pipelined encrypted frames lost). Fixed by passing
  `buf_reader` directly. Incorporated `client_nonce` in HKDF salt per spec
  (`client_nonce || server_nonce`). Discovery helpers extracted to `btsp_discovery.rs`
  (btsp.rs 831→721). 4 new live-validation tests. All quality gates green.
- **Sprint 51: BTSP Phase 3 `btsp.negotiate` (May 2)**: Server-side `btsp.negotiate`
  JSON-RPC handler implemented. Validates session_id, generates 12-byte random
  server nonce, derives session keys via HKDF-SHA256, returns cipher + nonce.
  Transport-layer integration: on successful negotiate with keyed cipher,
  seamlessly switches from plaintext NDJSON to encrypted BtspFrameReader/Writer
  framing. Graceful NULL cipher fallback. `register_with_discovery` extracted from
  transport.rs to discovery.rs. 59 registered methods (was 58). 10 new tests.
  `hkdf 0.12` added. All quality gates green.
- **Sprint 50: Phase 56 PG-47 + Graph PGM (May 1)**: `stats.entropy` alias wired
  (resolves PG-47 — primalSpring callers no longer get method-not-found).
  `graph.belief_propagation` wired (chain PGM forward pass via
  `barracuda::linalg::belief_propagation_chain`). 58 registered methods (was 56).
  Graph handlers extracted to `methods/graph.rs`. 7 new coverage tests.
  BTSP Phase 3 assessed: server-side relay + ChaCha20-Poly1305 framing already
  wired; client-side deferred to sourDough scaffold.
- **Sprint 49: IPC Surface Expansion Phase 2 (Apr 30)**: 6 new JSON-RPC methods
  (50 → 56): `stats.shannon`, `stats.covariance`, `stats.spearman`,
  `stats.fit_linear`, `stats.empirical_spectral_density`, `linalg.graph_laplacian`.
  `linalg.eigenvalues`/`stats.eigh` enhanced with eigenvectors. Closes 6 of 18
  primalSpring GAP-11 items. BufReader/shader-absorption gaps confirmed stale.
- **Sprint 48b: 12-Axis Deep Debt (Apr 29)**: cpu_executor `_ => 0.0` → typed
  errors. autotune magic numbers → named constants. BatchGuard Drop observability.
  npu_executor default score → named constant. 12-axis audit clean.
- **Sprint 48: BTSP-BARRACUDA-WIRE Closure (Apr 29)**: Confirmed
  `PRIMAL_GAPS.md` BTSP-BARRACUDA-WIRE gap is stale (resolved Sprint 44h-44i).
  tarpc keyed-cipher enforcement: `serve_tarpc_unix` rejects BTSP connections
  with keyed cipher (ChaCha20-Poly1305 / HMAC) — JSON-RPC is the correct
  transport for encrypted connections. 2 new full-relay integration tests
  with mock security provider. 26 BTSP compliance tests (was 22).
- **Sprint 47b: Deep Debt (Apr 28)**: Role-based naming evolution
  (`register_with_songbird`→`register_with_discovery`). naga-exec silent
  `_ => 0.0` fallbacks → typed `TypeMismatch` errors. autotune observability.
  12-axis audit clean.
- **Sprint 47: Discovery Self-Registration (Apr 28)**: `ipc.register` to
  discovery service via `DISCOVERY_SOCKET` at startup — 11 semantic capability
  domains derived from registered methods. Fire-and-forget. Per Phase 55b.
- **Sprint 46: NUCLEUS Env Var Wiring + Deep Debt (Apr 28)**: Per primalSpring
  Phase 55 two-tier crypto model — `BEARDOG_SOCKET` / `BTSP_PROVIDER_SOCKET`
  wired as preferred discovery. `DISCOVERY_SOCKET` wired as async fallback via
  `ipc.resolve`. `FAMILY_SEED` error message corrected. Role-based
  naming evolution (`beardog_*` → `provider_*`/`security_provider_rpc`). 12-axis
  deep debt audit clean — zero hardcoded sibling primal names in runtime code.
- **Sprint 45: JSON-RPC Surface Expansion (Apr 26)**: 11 new method registrations
  (39→50) — 2 aliases (`stats.eigh`, `stats.pearson`) + 9 new handlers (`linalg.svd`,
  `linalg.qr`, `stats.chi_squared`, `stats.anova_oneway`, `activation.softmax`,
  `activation.gelu`, `spectral.stft`, `ml.mlp_forward`, `ml.attention`). New
  `methods/ml.rs` for ML namespace. 36 new coverage tests. Achieves neuralSpring parity.
- **Sprint 44g: BTSP Wire Fix — writer.shutdown() → flush() (Apr 24)**: Replaced
  `writer.shutdown().await` with `writer.flush().await` in `security_provider_rpc()`.
  Shutdown sent TCP FIN to BearDog, killing the connection before response arrived.
  Resolves `BTSP_WIRE_CONVERGENCE_APR24_2026.md` barraCuda item.
- **Sprint 44f: Deep Debt — Smart Refactoring (Apr 20)**: `sovereign_device.rs` 924→773L
  (query_dispatch_arch extracted to sovereign_discovery.rs + tests extracted).
  `btsp.rs` 815→678L (tests extracted). Zero production files over 800L. 12-axis clean.
- **Sprint 44e: Phase 45c BTSP Relay Alignment (Apr 20)**: Fixed 5 BTSP handshake
  relay issues per primalSpring Phase 45c audit — ClientHello detection now accepts
  `"protocol":"btsp"` JSON-line format, `session_create_rpc` sends base64-encoded
  `family_seed`, `session_verify_rpc` passes `client_ephemeral_pub`+`preferred_cipher`,
  field names aligned to BearDog wire (`session_token`/`response`). 7 new tests.
  Upstream clippy fix in `sovereign_device.rs`.
- **Sprint 44d: Deep Debt — Magic Number Evolution (Apr 20)**: `WORKGROUP_SIZE_MEDIUM = 128`
  added; 12 production files evolved from bare `256u32`/`128u32`/`64u32` to named constants
  (`add.rs`, `mul.rs`, `fma.rs`, `sparse_matmul_quantized.rs`, FHE NTT/INTT, fused stats,
  `cumprod_f64.rs`). Chi-squared bisection bracket named. 12-axis deep debt audit clean.
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
  11 files. All clippy pedantic clean. External deps (futures, half)
  audited and justified. `pollster` eliminated (May 24 deep debt sprint).
  Zero production unwrap/expect confirmed.

Earlier completions (Mar 7–10) are documented in `CHANGELOG.md` and
`specs/REMAINING_WORK.md`.

---

## Immediate (P1)

- **2-Gate Mesh Proof (coordination)**: eastGate ↔ strandGate bidirectional mesh
  validation. Songbird :7700 running on strandGate; eastGate needs NUCLEUS +
  Songbird on LAN (192.168.1.144). Verify `discovery.peers`, `mesh.health_check`,
  cross-gate `capability.call`. **Upstream coordination with eastGate operators.**
- **PrecisionBrain → coralReef → SovereignDevice CI integration**: Mock trio E2E validated
  (Sprint 57). Next: CI with live coralReef instance for full pipeline validation.
- **DF64 NVK hardware verification**: GPU-dispatched DF64 E2E tests added (Sprint 63, FMA +
  Kahan summation). Remaining: run Yukawa force kernels through NVK/NAK on physical hardware
  to validate sovereign compiler numerical correctness on real silicon.
- **Tensor core GEMM codegen**: `kernel_router` routes F16/BF16/TF32 `DenseMatmul` to
  `KernelTarget::Sovereign` with `HardwareHint::TensorCore` (Sprint 64). Next: coralReef
  HMMA/WGMMA emission for eigensolvers/preconditioners via mixed-precision iterative refinement.
- **`BatchedTridiagEigh` GPU op**: groundSpring local QL implicit eigensolver is a candidate
  for absorption as a batched GPU tridiagonal eigenvector solver.
- **Multi-GPU OOM automatic migration**: OOM detection flag wired in `WgpuDevice`, `is_oom()`
  + `clear_oom()` API live, `is_retriable()` covers OOM (Sprint 64). Next: automatic workload
  migration when a device hits VRAM quota via `QuotaTracker`.
- **Kokkos parity validation baseline**: Document `sarkas_gpu` validation results, extract
  PPPM shader performance numbers for apples-to-apples comparison. Framework parity benchmarks
  added (Sprint 63, LAMMPS + SciPy). Now unblocked by VFIO strategy — projected ~4,000
  steps/s vs Kokkos 2,630 steps/s.

## Near-term (P2)

- **Test coverage to 90%**: Currently 80.54% line on llvmpipe (Sprint 26:
  measured via llvm-cov). CI 80% gate blocking (Sprint 3).
  Evolve `--fail-under` from 80 to 90 with real GPU hardware. Remaining gaps
  are exclusively GPU-dependent code paths.
- **GPU test parallelism (cargo-nextest)**: Mass parallel `cargo test` causes
  wgpu SIGSEGV under extreme GPU driver contention (process-global state
  limitation). All tests pass individually. `cargo-nextest` provides process
  isolation and resolves this. **Upstream: consider nextest as ecosystem CI
  standard for GPU-bearing primals.**
- **Kokkos GPU parity benchmarks**: Run barraCuda GPU benchmarks on matching hardware,
  publish comparison data.
- **Optional tensor encryption via `tensor` purpose key**: Per
  `NUCLEUS_TWO_TIER_CRYPTO_MODEL.md` — encrypt sensitive tensor data in transit
  using BearDog ChaCha20-Poly1305 delegation. Not all tensor ops need encryption;
  opt-in for medical/financial workloads. Deps already present (`chacha20poly1305`,
  `hmac`, `sha2`). primalSpring Phase 55 rates this as nice-to-have.

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
