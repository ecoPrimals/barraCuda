# barraCuda

**Version**: 0.3.12
**Status**: Standalone primal — zero cross-dependencies, fully concurrent, all quality gates passing
**License**: AGPL-3.0-or-later (scyBorg provenance trio)
**MSRV**: 1.87

---

## The Name

**barraCuda** — *BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra*

More concept than exact acronym. The barracuda stands still until it strikes —
fast, silent, instant. That's the compute model: sovereign math that waits on
any GPU, then executes instantly across any vendor's silicon.

---

## What is barraCuda?

barraCuda is the **sovereign math engine** for the ecoPrimals ecosystem. It
provides GPU-accelerated scientific computing across any vendor's hardware
using WGSL shaders compiled through wgpu. One source, any GPU, identical
results.

### Key capabilities

- **826 WGSL shaders** spanning scientific compute domains (all with SPDX license headers)
- **1,116 Rust source files**, 25 integration test harnesses, 4,393+ tests passing via nextest CI profile
- **DF64 emulation** — double-precision arithmetic on GPUs without native f64
- **FHE on GPU** — Number Theoretic Transform, INTT, pointwise modular
  multiplication via 32-bit emulation of 64-bit modular arithmetic. The only
  cross-vendor FHE GPU implementation in existence.
- **Lattice QCD** — SU(3) gauge theory, staggered Dirac, CG solver, HMC, GPU multi-shift CG (Jegerlehner), GPU-resident RHMC observables (Hamiltonian/Metropolis/fermion action)
- **Spectral analysis** — Anderson localization, Lanczos eigensolver with Ritz eigenvector extraction
- **Molecular dynamics** — Yukawa, PPPM, VV integrator, cell-list neighbor search
- **Linear algebra** — dense, sparse (CSR SpMV, CG, BiCGStab), eigensolvers, L-BFGS
- **Statistics** — bootstrap, jackknife, diversity indices, hydrology
- **Pharmacometrics** — FOCE gradients, VPC Monte Carlo, population PK, dose-response
- **Health** — Michaelis-Menten PK (GPU batch), SCFA production, beat classification, EDA stress detection
- **Bioinformatics** — Smith-Waterman, HMM, phylogenetics, bipartition encoding, genomic ops
- **ML ops** — matmul, softmax, attention, ESN reservoir computing
- **Sovereign shader compilation** — naga 28 IR optimizer, SPIR-V passthrough
- **NagaExecutor** — CPU interpreter for naga IR, executes WGSL compute shaders without GPU (f32+f64 native, shared memory, barriers, atomics)
- **coralReef IPC contract** — sovereign CPU compilation (`shader.compile.cpu`, `shader.execute.cpu`) and validation (`shader.validate`) via JSON-RPC
- **JSON-RPC 2.0 + tarpc** — dual-protocol IPC with 50 bare semantic `{domain}.{operation}` methods; Wire Standard L2 compliant (`{primal, version, methods}` envelope, `identity.get`, `provided_capabilities`)
- **UniBin CLI** — single `barracuda` binary with `server --port <PORT>`, `service`, `doctor`, `validate`, `version`

### Design principles

1. **Math is universal, precision is silicon** — one WGSL source, any precision
2. **Vendor-agnostic** — same binary, identical results on any GPU
3. **Sovereign** — zero external SDK dependency for correctness or performance
4. **Pure Rust** — `#![forbid(unsafe_code)]` in both crates, zero `unsafe` blocks, zero external C dependencies, zero dependencies on any other primal (lifecycle and health traits internalized from sourDough scaffold)
5. **Fully concurrent** — `GuardedDeviceHandle` + atomic encoder barrier prevents wgpu-core races without lock contention; split-lock GPU submission (submit and poll use separate lock acquisitions); fire-and-forget dispatch via `submit_commands` for non-readback ops; wgpu 28 `Device`/`Queue` are `Clone` — zero `Arc` overhead for handle sharing; all tests pass at 16 threads on llvmpipe
6. **AGPL-3.0** — free as in freedom

---

## Recent

- **Sprint 47b: Deep Debt (Apr 28)**: Role-based naming (`register_with_songbird`→`register_with_discovery`, `songbird_capability_domains`→`discovery_capability_domains`). naga-exec silent `_ => 0.0` fallbacks → typed errors. autotune observability. 12-axis audit clean.
- **Sprint 47: Discovery Self-Registration (Apr 28)**: `ipc.register` to discovery service via `DISCOVERY_SOCKET` at startup — 11 capability domains derived from `REGISTERED_METHODS`. Fire-and-forget. Per Phase 55b.
- **Sprint 46: NUCLEUS Env Var Wiring + Deep Debt (Apr 28)**: Per Phase 55 two-tier crypto model — `BEARDOG_SOCKET`/`BTSP_PROVIDER_SOCKET` wired as preferred discovery, `DISCOVERY_SOCKET` wired as async `ipc.resolve` fallback, `FAMILY_SEED` error message corrected. Role-based naming evolution (`beardog_*` → `provider_*`/`security_provider_rpc`). 12-axis deep debt audit clean.
- **Sprint 45/45b: JSON-RPC Surface Expansion + Deep Debt (Apr 26)**: 11 new method registrations (39→50) for neuralSpring parity — `linalg.svd`, `linalg.qr`, `stats.chi_squared`, `stats.anova_oneway`, `activation.softmax`, `activation.gelu`, `spectral.stft`, `ml.mlp_forward`, `ml.attention` + 2 aliases (`stats.eigh`, `stats.pearson`). New `methods/ml.rs` and `methods/spectral.rs` modules. `math.rs` smart-refactored (819→641L). Shared `params.rs` eliminates DRY violation. 36 new coverage tests. 12-axis deep debt audit clean.
- **Sprint 44g: BTSP Wire Fix + 12-Axis Audit (Apr 24)**: `security_provider_rpc()` `writer.shutdown()` → `writer.flush()` — fixes BearDog connection loss. 12-axis deep debt audit clean bill. 4,393+ tests, all quality gates green.
- **Sprint 44f: Smart Refactoring (Apr 20)**: `sovereign_device.rs` 924→773L, `btsp.rs` 815→678L. Zero production files >800L.
- **Sprint 44e: BTSP Relay Alignment (Apr 20)**: 5 BTSP handshake relay fixes per Phase 45c. 7 new tests.
- **Sprint 44d: Magic Number Evolution (Apr 20)**: 12 files evolved from bare workgroup size literals to named constants.
- **Sprint 44c: CPU Tensor Fallback (Apr 20)**: Handle-based tensor ops (`create/matmul/add/scale/clamp/reduce/sigmoid`) work on headless hosts via `CpuTensor` store.
- **Sprint 44: Composition Audit (Apr 20)**: 7 new JSON-RPC methods (32→39), Fitts' law corrected, response schema standardized, `tensor.matmul_inline`.
- **Sprint 40–43**: SovereignDevice 3-tier fallback, cpu-shader default-on, Docker bind resolution, deep debt evolution, 826/826 WGSL SPDX headers, BTSP Phase 3, FAMILY_ID scoping, capability-based discovery.
- **Sprint 39: primalSpring Audit Remediation (Apr 10)**: BTSP Phase 2 full handshake — `guard_connection()` evolved to 6-step X25519+HMAC relay (ClientHello/ServerHello/ChallengeResponse/HandshakeComplete) with legacy fallback. BC-GPU-PANIC fixed — `Auto::new()` decoupled from test pool, graceful CPU-only degradation. fault_injection SIGSEGV — `gpu-serial` added to `stress`/`gpu` profiles. Musl rebuild: fresh binaries with checksums. 4,422 tests pass, all quality gates green.
- **Sprint 38: Deep Debt — BTSP Phase 2, Capability-Based Discovery & Idiom Sweep (Apr 9)**: BTSP Phase 2 connection authentication guard integrated into all accept loops (`serve_unix`/`serve_tcp`/`serve_tarpc_unix`). BearDog discovery evolved from hardcoded `beardog-core.json` to capability-based `discover_by_capability()` — scans all `*.json` discovery files for `btsp.session.create` method. `Box<dyn Error>` → typed `BarracudaCoreError::ipc()`. `#[allow]` → `#[expect]` with reason. `precision_brain.rs` smart-refactored (703→421 LOC). 4 GPU test binaries serialized. Musl-static rebuild fixed (static-pie). 4,421 tests pass, all quality gates green.
- **Sprint 37: Deep Debt — Test Module Refactor & Code Cleanup (Apr 8)**: `methods_tests.rs` (951 LOC) smart-refactored into 6 domain-focused test modules + hub (largest module 193 lines). `buffer_test.rs` println! noise removed. `nadam_gpu.rs` stale evolution comment removed. `force_interpolation.rs` indexed loop → idiomatic iterator. 12-axis deep debt audit: clean bill. Zero files >800L. 4,207 tests pass, all quality gates green.
- **Sprint 36: Domain-Based Socket Naming & Flaky Test Serialization (Apr 8)**: Socket naming evolved from primal-based (`barracuda.sock`) to domain-based (`math.sock` / `math-{fid}.sock`) per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3. Legacy `barracuda.sock` symlink for backward compatibility. New `PRIMAL_DOMAIN` constant. `identity.get` and `primal.capabilities` domain field evolved from `"compute"` to `"math"`. `three_springs_tests` added to `gpu-serial` nextest group (Mesa llvmpipe SIGSEGV mitigation). 4,207 tests pass, all quality gates green.
- **Sprint 35: Deep Debt — Typed Errors, thiserror, Transport Refactor (Apr 8)**: `validate_insecure_guard` evolved from `Result<(), String>` to typed `BarracudaCoreError::Lifecycle`. `PppmError` manual `Display`+`Error` evolved to `#[derive(thiserror::Error)]`. `transport.rs` (866 LOC) smart-refactored: test module extracted to `transport_tests.rs`, domain socket naming + legacy symlink added (production file now 528 LOC). 12-axis deep debt audit: clean bill on all axes. 4,207 tests pass, all quality gates green.
- **Sprint 34: BTSP Socket Naming + BIOMEOS_INSECURE Guard (Apr 8)**: Resolves GAP-MATRIX-12 — `FAMILY_ID` socket scoping with standard env var precedence (`BARRACUDA_FAMILY_ID` → `FAMILY_ID` → `BIOMEOS_FAMILY_ID`), `BIOMEOS_SOCKET_DIR` env var support, `BIOMEOS_INSECURE` guard (refuse to start when both `FAMILY_ID` and `BIOMEOS_INSECURE=1`). Per `BTSP_PROTOCOL_STANDARD.md` §Compliance and `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3-4. GAP-MATRIX-06 plasmidBin metadata updated to v0.3.11 with full capability manifest. 20 new BTSP compliance tests (`btsp_socket_compliance.rs`). 4,207 tests pass, all quality gates green.
- **Sprint 33: Wire Standard L2 Compliance (Apr 8)**: `capabilities.list` now returns Wire Standard L2 `{primal, version, methods}` envelope with `provided_capabilities` grouping, `consumed_capabilities`, `protocol`, `transport`. New `identity.get` method returns `{primal, version, domain, license}`. Both JSON-RPC and tarpc paths wired. 31 methods (was 30). `provided_capability_groups()` in discovery module derives structured groups from the dispatch table — zero hardcoded domain catalog. 13 new L2 compliance tests. 4,187 tests pass, all quality gates green.
- **Sprint 32: Fault Injection SIGSEGV Resolution & Deep Debt Audit (Apr 7)**: Root-caused Mesa llvmpipe within-process thread safety SIGSEGV in 3 fault injection tests. Serialized concurrent GPU readbacks in `fault_concurrent_tensor_access` and `test_concurrent_error_handling`. Bounded `fault_out_of_gpu_memory` allocation loop (10,000→256 iterations). Updated nextest coverage profile from deprecated `exclude = true` to `default-filter` syntax. Fixed 4 clippy findings: non-existent `needless_type_cast` lint, protocol string `"jsonrpc-2.0"` → `"json-rpc-2.0"`, 2 unfulfilled `dead_code` expects. Comprehensive 12-axis deep debt audit: zero production unsafe/unwrap/expect/println/mocks/hardcoding/TODO/commented-out code, zero `#[allow(`, zero `Result<T,String>` in production, zero files >1000 lines. 4,180 tests pass (CI profile), 0 failures. All quality gates green.
- **Sprint 31: Deep Debt Cleanup & Test Stability Hardening (Apr 5)**: Removed deprecated `CoralReefDevice` alias (zero consumers). Evolved `SpirvError` to thiserror derive. Fixed 12 misleading dead_code reason strings on GPU API impls. Gated 11 additional SIGSEGV-prone test binaries behind `stress-tests` feature — `cargo test --workspace` now 100% clean. Comprehensive deep debt audit: zero production unwrap/expect/panic, zero hardcoded primal names, zero mocks in production, zero TODO/FIXME, all files under 845 lines. All quality gates green.
- **Sprint 30: Deep Debt Audit, Smart Refactoring & Test Stability (Apr 5)**: Smart refactor of `executor.rs` (934→208 lines) + new `invocation.rs` (756 lines) — `DispatchCoords` struct eliminates `too_many_arguments`. SIGSEGV fix via nextest `gpu-serial` test group (chaos/fault/property tests serialized). Disabled `test_nn_vision_integration` evolved to `test_vision_pipeline_preprocessing` (8/8 integration tests pass). All quality gates green.
- **Sprint 29: Deep Debt Cleanup & Shader-First Evolution (Apr 4)**: Unified magic `256` workgroup size → `WORKGROUP_SIZE_1D` constant across 15+ files (shader_dispatch, jackknife, biosignal, gradient, cpu_executor, perlin_noise, population_pk, hill_dose_response, michaelis_menten_batch, scfa_batch, beat_classify, rop_force_accum). Removed unused `num-traits` from workspace. Smart refactor of `executor.rs` (1,097→932 lines, vector ops extracted to `vector_ops.rs`). `eval_math` decomposed into 4 focused functions (eval.rs 629→527 lines, `too_many_lines` suppression eliminated). Production `expect()` in `wgpu_backend.rs` evolved to safe pattern-match + `Result`. Misleading `nautilus/readout.rs` "no-op" doc corrected. `coralReef` doc references evolved to capability-based discovery language. `"biomeos"` / `"ecoPrimals"` namespace strings consolidated into shared constants. Perlin noise 7× `#[expect]` blocks consolidated to 2 helper functions. All quality gates green (3,815 lib + 16 naga-exec tests, 0 failures).
- **Sprint 27**: primalSpring downstream audit remediation — hex bitwise literal (`0x3D`), `#[expect]` reason strings, barracuda-core lint promotions (`use_self`/`map_unwrap_or` → warn). All clippy/fmt/deny/doc gates green. 4,600+ tests, zero debt markers.
- **Sprint 26**: Comprehensive audit, executor refactor, cargo deny fix — WorkgroupMemory subsystem extracted (executor.rs 1,020→886 lines). Stale `#[allow]` removed, `#[allow(unused_async)]` → `#[expect]` in core. Full audit confirmed zero production unwrap/panic/expect. 80.54% coverage.
- **Sprint 25**: Deep debt evolution — zero panics, modern idiomatic Rust, capability-based naming across all production code.
- **Sprint 24**: WGSL-as-truth test architecture + NagaExecutor + coralReef sovereign compilation — Migrated 337 GPU op test files from `get_test_device_if_gpu_available()` to `get_test_device()`, enabling 2,770 tests to run on CPU/llvmpipe (was ~0 coverage on CI). 17 GPU-exclusive modules correctly identified and re-gated. New crate `barracuda-naga-exec`: pure-Rust CPU interpreter for naga IR with f32/f64 native support, workgroup shared memory, barriers, atomics (16 tests). `assert_shader_math!` and `assert_shader_math_f64!` macros for zero-GPU shader validation. coralReef IPC contract: 10 new wire types, 5 new `CoralCompiler` methods (`compile_cpu`, `execute_cpu`, `validate_shader`), capability discovery for `shader.compile.cpu` and `shader.validate`. `ShaderValidationBackend` enum with coralReef-first fallback chain. 4-layer validation architecture (llvmpipe / NagaExecutor / coralReef CPU / real GPU). All quality gates green. 2,786 total tests, 0 failures.
- **Sprint 23**: ludoSpring V35 gap resolution — 15 new IPC methods wired (math.sigmoid, math.log2, stats.mean, stats.std_dev, stats.weighted_mean, noise.perlin2d, noise.perlin3d, rng.uniform, activation.fitts, activation.hick, tensor.add, tensor.scale, tensor.clamp, tensor.reduce, tensor.sigmoid). 30 total JSON-RPC methods. Socket path fixed to `barracuda.sock` per PRIMAL_IPC_PROTOCOL. Dual-transport startup (UDS + TCP via `BARRACUDA_PORT` env var). All `#[allow(` migrated to `#[expect(` or `cfg_attr` in both crates. Release binary 4.7MB. 3,808 tests, all quality gates green.
- **Sprint 22**: Spring absorption & deep debt evolution — Critical fermion force sign fix (neg_eta convention) in 2 staggered/pseudofermion WGSL shaders. 8 WGSL shaders absorbed from hotSpring: 5 multi-shift CG (Jegerlehner zeta, shifted alpha/xr/x/p) + 3 GPU-resident (Hamiltonian assembly, fermion action, Metropolis). `gpu_multi_shift_cg.rs` orchestration with generic CPU reference. `gpu_resident_observables.rs` with O(1)-readback pipelines. 6 RHMC/lattice tolerance constants (42 total). f32 Perlin 2D shader + API for ludoSpring. 32-bit LCG contract for ludoSpring. Lanczos eigenvector pipeline with Ritz vector Q×z back-transform for groundSpring. 824 WGSL shaders, all quality gates green.
- **Sprint 21**: Compliance & coverage deep evolution — `health.liveness`, `health.readiness`, `capabilities.list` endpoints implemented per wateringHole Semantic Method Naming Standard v2.2.0 with all required aliases (`ping`, `health`, `status`, `check`, `capability.list`). Validation-first handler refactoring across JSON-RPC and tarpc layers (validate inputs before device check). `--port` CLI flag per UniBin standard. `barracuda-spirv` unsafe code evolved to `#![deny(unsafe_code)]` + targeted `#[allow]`. barracuda-core coverage 59.33% → 72.83% line (+13.5pp), 214 unit tests + 8 e2e (up from 148). rpc.rs refactored to extract tests (861→572 lines). All quality gates green.
- **Sprint 20**: FMA evolution & lint promotion — 625 `suboptimal_flops` sites evolved to `mul_add()` for hardware FMA precision. 4 lints promoted from `allow` to `warn`: `suboptimal_flops` (415→0), `use_self` (332→0), `tuple_array_conversions` (2→0), `needless_range_loop` (45→0). All `needless_range_loop` sites evolved to idiomatic iterators. 232 files changed, 3,623+ tests pass, zero clippy errors.
- **Sprint 19**: Deep debt solutions & idiomatic Rust evolution — RPC `tolerances_get` evolved to centralized tolerance registry. Cast safety: all `usize as u32` in `TensorSession` replaced with checked casts. 6 domain feature gates added (`domain-fhe`, `domain-md`, `domain-lattice`, `domain-physics`, `domain-pharma`, `domain-genomics`). `FlatTree::validate()` evolved to typed errors.
- **Sprint 18**: Ecosystem absorption & API housekeeping — full pull + review of 8 springs + 10 primals. `GpuDriverProfile` removed. `barracuda::cast` module with safe numeric casts. ESN device accessors. f64 shader constants exposed. Tolerance stability contract.
- **Sprint 17**: Nursery linting, IPC naming evolution & coverage push — `clippy::nursery` blanket-enabled, IPC method names evolved to bare `{domain}.{operation}`, 13 pooling tests hardened, coverage 71.59% line / 78.44% function.

---

## Deployment Modes and GPU Constraints

| Deployment | GPU Path | CPU Shader | Sovereign IPC |
|------------|----------|------------|---------------|
| **glibc host** (desktop, server) | wgpu Vulkan/Metal/DX12 | Yes (default) | Optional |
| **musl-static** (ecoBin, Alpine, Docker) | **Unavailable** — `dlopen` cannot load GPU drivers | Yes (default) | **Yes** — GPU via IPC to coralReef+toadStool |
| **WASM** | wgpu WebGPU | Not yet | No |

**Why musl-static cannot use GPU directly:** wgpu requires `dlopen` to load
Vulkan/Mesa driver shared objects at runtime. musl-static binaries are fully
statically linked — there are no `.so` files to load. This is a fundamental
constraint of static linking, not a barraCuda bug.

**Solution:** ecoBin musl-static binaries use the cpu-shader path (now default,
BC-08) for standalone compute, or sovereign IPC dispatch (BC-07) to delegate
GPU work to a coralReef+toadStool peer running on a glibc host with GPU access.
`Auto::new()` handles this automatically: wgpu GPU → wgpu CPU → Sovereign IPC → Err.

---

## Architecture

barraCuda is a 4-crate workspace:

- **`barracuda`** — the math engine (826 WGSL shaders, 15-tier precision, all GPU ops)
- **`barracuda-core`** — primal lifecycle (JSON-RPC, tarpc, UniBin CLI)
- **`barracuda-spirv`** — SPIR-V passthrough bridge (isolates the single `unsafe` call)
- **`barracuda-naga-exec`** — CPU interpreter for naga IR (shader-first CPU execution + GPU validation)

Springs and other consumers `cargo add barracuda`. toadStool orchestrates above
it; barraCuda owns the math.

```
Your Code / Springs
    |
    v
barracuda (umbrella crate)
    |-- Pure Math: linalg, special, numerical, spectral, stats, sample, activations, rng, tolerances
    |-- GPU Math: ops (bio, pharma, fhe, qcd, ...), tensor, shaders, interpolate, optimize
    |-- Compute Fabric: device, staging, pipeline, dispatch, multi_gpu
    |-- Domain Models: nn, snn, esn, pde, genomics (feature-gated)
    |
    v
barracuda-core (primal lifecycle)
    |-- IPC: JSON-RPC 2.0 (text) + tarpc (binary)
    |-- UniBin CLI: server, doctor, validate, version
    |-- lifecycle + health: PrimalLifecycle, PrimalHealth (owned)
    |
    v
wgpu 28 (WebGPU)
    |
    +-- Vulkan (NVIDIA, AMD, Intel)
    +-- Metal (Apple)
    +-- DX12 (Windows)
    +-- Software rasterizer (CPU fallback / CI)
```

See `specs/ARCHITECTURE_DEMARCATION.md` for the full barraCuda / toadStool
boundary definition.

---

## Structure

```
barraCuda/
├── Cargo.toml                       # Workspace manifest
├── deny.toml                        # cargo-deny (license + advisory audit)
├── rustfmt.toml                     # Formatting config
├── README.md                        # You are here
├── CONTRIBUTING.md                  # How to contribute
├── CONVENTIONS.md                   # Coding standards
├── CHANGELOG.md                     # SemVer changelog
├── BREAKING_CHANGES.md              # Migration notes
├── STATUS.md                        # Quality scorecard (A+ grade)
├── WHATS_NEXT.md                    # P1–P4 roadmap
├── START_HERE.md                    # Developer quick start
├── PURE_RUST_EVOLUTION.md           # Sovereign compute evolution log
├── SPRING_ABSORPTION.md             # Cross-spring absorption tracker
├── LICENSE                          # AGPL-3.0-or-later (scyBorg trio)
├── .github/workflows/ci.yml        # CI: fmt, clippy, deny, doc, test, coverage
├── crates/
│   ├── barracuda-core/              # Primal lifecycle wrapper
│   │   ├── src/lib.rs               # BarraCudaPrimal: start/stop/health
│   │   ├── src/ipc/                 # JSON-RPC 2.0 server + transport (50 methods, Wire Standard L2)
│   │   ├── src/rpc.rs               # tarpc service definition (16 endpoints, parity with JSON-RPC)
│   │   └── src/bin/barracuda.rs     # UniBin CLI
│   └── barracuda/                   # Umbrella crate — all math + GPU
│       ├── src/
│       │   ├── lib.rs               # Module declarations + prelude
│       │   ├── error.rs             # BarracudaError (19 variants incl. DeviceLost)
│       │   ├── linalg/              # Dense/sparse linear algebra
│       │   ├── special/             # Special functions (erf, gamma, Bessel)
│       │   ├── numerical/           # Gradients, integration, ODE
│       │   ├── spectral/            # Anderson, Lanczos, eigensolve
│       │   ├── stats/               # Bootstrap, regression, distributions
│       │   ├── sample/              # LHS, Sobol, Metropolis, sparsity
│       │   ├── ops/                 # GPU ops (matmul, softmax, FHE, bio)
│       │   ├── tensor/              # GPU tensor type
│       │   ├── shaders/             # 826 WGSL shaders (see shaders/README.md)
│       │   ├── device/              # GpuBackend trait, WgpuDevice, SovereignDevice, concurrency
│       │   ├── staging/             # Ring buffers, unidirectional pipelines
│       │   ├── pipeline/            # ComputeDispatch, batched pipelines
│       │   ├── dispatch/            # Size-based CPU/GPU routing
│       │   ├── multi_gpu/           # GpuPool, MultiDevicePool, load balancing
│       │   ├── unified_hardware/    # Unified CPU/GPU/NPU abstraction
│       │   └── ...                  # + nn, snn, esn, pde, genomics, vision
│       ├── examples/                # Runnable examples
│       ├── tests/                   # 42 integration test files (25 harnesses + submodules)
│       └── src/bin/                 # validate_gpu, bench_*
└── specs/
    ├── BARRACUDA_SPECIFICATION.md       # Crate architecture + IPC contract
    ├── PRECISION_TIERS_SPECIFICATION.md # 15-tier precision ladder (Binary→DF128)
    ├── REMAINING_WORK.md                # P1-P4 open work items
    └── ARCHITECTURE_DEMARCATION.md      # barraCuda vs toadStool boundaries
```

---

## Concurrency Model

barraCuda's GPU access uses a three-layer concurrency model that prevents
wgpu-core internal races without lock contention:

1. **`active_encoders: AtomicU32`** — lock-free counter incremented during
   any wgpu-core activity (buffer creation, shader compilation, command
   encoding). Multiple threads increment simultaneously with zero contention.
2. **`gpu_lock: Mutex<()>`** — serializes `queue.submit()` and `device.poll()`
   against each other. Submit and poll use **separate** lock acquisitions so
   other threads can interleave submits while one thread polls. Before poll,
   a bounded yield loop waits for active encoders to reach zero (encoding is
   CPU-speed microsecond work).
3. **`dispatch_semaphore`** — hardware-aware cap (2 for CPU/llvmpipe, 8 for
   discrete GPU) preventing driver overload.

`GuardedEncoder` is an RAII wrapper that auto-decrements the active encoder
count on finish or drop, making the barrier leak-proof.

---

## Quality Gates

```bash
cargo fmt --all -- --check              # formatting
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lints (pedantic, all clean)
cargo deny check                        # license + advisory audit
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps  # documentation (zero warnings)
cargo build --workspace                 # compilation
cargo nextest run --workspace --profile ci  # 4,393+ tests via nextest
cargo llvm-cov --workspace --lib        # 80% CI gate (blocking), 90% target (requires GPU hardware)
```

All gates are enforced in `.github/workflows/ci.yml`.

---

## IPC Protocol

barraCuda exposes a dual-protocol IPC interface per wateringHole standards:

**JSON-RPC 2.0** (primary, text, newline-delimited TCP/Unix socket):

| Method | Description |
|--------|-------------|
| `identity.get` | Wire Standard L2 identity: `{primal, version, domain, license}` |
| `primal.info` | Primal identity (name, version, protocol, namespace, license) |
| `primal.capabilities` | Advertise capabilities, methods, hardware state |
| `capabilities.list` | Ecosystem-standard capability probe (alias for `primal.capabilities`) |
| `device.list` | List available compute devices |
| `device.probe` | Probe device capabilities and limits |
| `health.liveness` | Fast liveness probe (aliases: `ping`, `health`) |
| `health.readiness` | Readiness probe — can the primal serve requests? |
| `health.check` | Full health check (aliases: `status`, `check`) |
| `tolerances.get` | Numerical tolerances for a named operation |
| `validate.gpu_stack` | GPU validation suite |
| `compute.dispatch` | Dispatch a named compute operation (zeros, ones, read) |
| `math.*` / `stats.*` | `math.sigmoid`, `math.log2`, `stats.mean`, `stats.std_dev`, `stats.variance`, `stats.correlation`, `stats.pearson`, `stats.weighted_mean`, `stats.chi_squared`, `stats.anova_oneway`, `stats.eigh` |
| `linalg.*` | `linalg.solve`, `linalg.eigenvalues`, `linalg.svd`, `linalg.qr` |
| `spectral.*` | `spectral.fft`, `spectral.power_spectrum`, `spectral.stft` |
| `noise.*` / `rng.*` | `noise.perlin2d`, `noise.perlin3d`, `rng.uniform` |
| `activation.*` | `activation.fitts`, `activation.hick`, `activation.softmax`, `activation.gelu` |
| `ml.*` | `ml.mlp_forward`, `ml.attention` |
| `tensor.*` | `tensor.create`, `matmul`, `matmul_inline`, `add`, `scale`, `clamp`, `reduce`, `sigmoid` |
| `fhe.*` | `fhe.ntt`, `fhe.pointwise_mul` |

50 methods follow the wateringHole `{domain}.{operation}` Semantic Method Naming
Standard v2.2.0. Wire Standard L2 compliant: `capabilities.list` returns the
`{primal, version, methods}` envelope with `provided_capabilities` grouping.
`health.liveness`, `health.readiness`, `health.check`, and `capabilities.list`
are non-negotiable ecosystem probes. Legacy `barracuda.{domain}.{operation}`
format accepted for backward compatibility.

**tarpc** (optional, binary, high-throughput primal-to-primal):

Same 16 endpoints with strongly-typed Rust signatures and full parameter
parity with the JSON-RPC handlers. Enabled via
`barracuda server --tarpc-bind 127.0.0.1:9001`.

---

## UniBin CLI

```bash
# Start IPC server (JSON-RPC on TCP)
barracuda server --port 9000
barracuda server --bind 127.0.0.1:9000

# Start with tarpc alongside JSON-RPC
barracuda server --bind 127.0.0.1:9000 --tarpc-bind 127.0.0.1:9001

# Start with Unix socket (default: $BIOMEOS_SOCKET_DIR/math.sock)
barracuda server --unix

# Start as systemd/init service (genomeBin mode)
barracuda service

# Health check and device diagnostics
barracuda doctor

# GPU validation suite
barracuda validate
barracuda validate --extended

# Version info
barracuda version
```

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu` | Yes | GPU compute via wgpu/WGSL. |
| `cpu-shader` | Yes | CPU WGSL interpreter via barracuda-naga-exec. Enables ecoBin compute without wgpu. |
| `domain-models` | Yes | All domain modules (nn, snn, esn, pde, genomics, vision, timeseries). |
| `domain-nn` | via umbrella | Neural network training API. |
| `domain-snn` | via umbrella | Spiking neural networks. |
| `domain-esn` | via umbrella | Echo state networks (reservoir computing). |
| `domain-pde` | via umbrella | PDE solvers (Richards, Crank-Nicolson). |
| `domain-genomics` | via umbrella | Bioinformatics and genomics API. |
| `domain-vision` | via umbrella | Computer vision pipelines. |
| `domain-timeseries` | via umbrella | Time series analysis (implies `domain-esn`). |
| `sovereign-dispatch` | No | Sovereign GPU dispatch via IPC to coralReef + toadStool (bypasses wgpu/Vulkan). |
| `serde` | No | Serde derive support. |
| `parallel` | No | Rayon parallelism hints. |

### Common dependency configurations

```toml
# Full (default) — everything
barracuda = { path = "../barraCuda/crates/barracuda" }

# Math + GPU only — no domain models (fastest compile)
barracuda = { path = "../barraCuda/crates/barracuda", default-features = false, features = ["gpu"] }

# Pure CPU math — no GPU at all (sub-2s compile)
barracuda = { path = "../barraCuda/crates/barracuda", default-features = false }
```

---

## Development Setup

### Prerequisites

- **Rust 1.87+** (`rustup update stable`)
- **GPU drivers** (Vulkan-capable: NVIDIA 525+, Mesa 23+, or Apple Metal)
- **llvmpipe** (optional, for headless CI — `sudo apt install mesa-vulkan-drivers`)
- **cargo-deny** (`cargo install cargo-deny`)
- **cargo-llvm-cov** (`cargo install cargo-llvm-cov`, for coverage)

### Sibling repositories

```
ecoPrimals/
├── barraCuda/          # This repo
├── coralReef/          # Shader compiler (WGSL/SPIR-V → native GPU binary)
├── sourDough/          # Scaffold reference (no runtime dependency)
├── toadStool/          # Orchestration / hardware discovery
├── wateringHole/       # Ecosystem standards and genomeBin manifest
└── ...Springs          # Consumers
```

### Build and test

```bash
cargo build --workspace
cargo test --workspace --lib
cargo run -p barracuda-core --bin barracuda -- doctor
```

---

## Relationship to ecoPrimals

barraCuda is a **NUCLEUS foundation primal** registered in `wateringHole/genomeBin/manifest.toml`.

- **toadStool** (orchestration) — routes compute to the best hardware. barraCuda
  is the execution layer; toadStool is the orchestration layer.
- **bearDog** (crypto) — FHE key generation + barraCuda GPU compute = sovereign
  encrypted computation.
- **songBird** (network) — toadStool uses songBird for multi-node distribution.
  barraCuda does not depend on songBird directly.
- **Springs** (validation) — domain-specific projects that consume barraCuda.

### Dependency direction

```
Springs ──> barraCuda (direct cargo dep)
toadStool ──> barraCuda (as compute backend)
bearDog ··> barraCuda (for FHE math)
barraCuda ──> (standalone — lifecycle/health internalized from sourDough scaffold)
```

barraCuda has ZERO dependencies on toadStool, songBird, bearDog, nestGate, or sourDough.
Lifecycle and health traits are modeled on the ecoPrimals pattern but fully owned by barraCuda.

---

## Specs

| Document | Purpose |
|----------|---------|
| `specs/BARRACUDA_SPECIFICATION.md` | Crate architecture, IPC contract, shader pipeline |
| `specs/PRECISION_TIERS_SPECIFICATION.md` | Full 15-tier precision ladder (Binary to DF128) |
| `specs/ARCHITECTURE_DEMARCATION.md` | barraCuda vs toadStool boundary definition |
| `specs/REMAINING_WORK.md` | P1-P4 open work items |
| `specs/GPU_SILICON_CAPABILITY_MATRIX.md` | GPU hardware FP64 rates, DF64 strategy, silicon exposure |
| `SOVEREIGN_PIPELINE_TRACKER.md` | Sovereign pipeline tracker (P0 blocker, libc evolution, cross-compilation) |
| `crates/barracuda/src/shaders/README.md` | Shader organization |

---

**Created with sourDough. Budded from toadStool S88-S89. Evolved to standalone primal.**
