# barraCuda — Evolution Record

The changelog is sorted by capability surface, not by date.
Each section tells the arc: what emerged, what it replaced, what it became.
The date-sorted history lives in `CHANGELOG_ARCHIVE.md`.

`H(data|epitope) < H(data|date) < H(data)`

---

## GPU Compute & WGSL Shaders — sovereign shader execution

Compute began as CPU-only math stubs; it became a WGSL-as-truth GPU pipeline with f32/f64 precision ladders, SPIR-V passthrough, and graceful GPU-less survival.

- **WGSL-as-truth pipeline** (Sprint 24→Wave 120): NagaExecutor + coralReef CPU compilation path; WGSL source is canonical, PTX/SPIR-V derived.
- **f64 transcendental pipeline** (Sprint 22d→Wave 155n): `SHADER_F64` native on ironGate RTX 5070; diversity shader self-recursion fix; subgroup reduction entry point corrected for Blackwell SM100+.
- **Graceful GPU-less survival** (Wave 54): `--no-gpu-probe` / `BARRACUDA_NO_GPU_PROBE` — instant cpu-shader-only startup on GPU-less hosts.
- **GPU buffer alignment** (Wave 156k): `aligned_copy_from_mapped<T>()` fixes `bytemuck::cast_slice` panic on unaligned GPU mapped memory.
- **ironGate operational** (Wave 120→123): RTX 5070 live via glibc build; 14/9 f64 builtins native; systemd override to local release binary.
- **Showcase fossilization** (Wave 49): Demo collection archived to fossilRecord; plasmidBin sole binary distribution channel.

---

## IPC Surface & Neural API — 101 methods of scientific compute

The RPC surface grew from scattered math helpers to 101 registered JSON-RPC methods with Neural API self-announcement and method gate compliance.

- **Method count evolution** (Sprint 45→Wave 156p): 87 → 90 → 98 → 101 methods including `protocols.list`, `device.video_codecs`, `linalg.batched_tridiag_eigh`.
- **Neural API announce** (Wave 43→44): Outbound `primal.announce` with socket field, signal tiers, cost/latency hints; capabilities narrowed to `["math", "shader", "compute"]`.
- **G65 protocol negotiation** (Wave 156p): Single-socket tarpc/JSON-RPC selection; `TransportStream::peek()` via safe `rustix::net::recv(PEEK)` — zero unsafe in barracuda-core.
- **Method gate JH-0** (Sprint 54): Pre-dispatch capability gate shipped; barraCuda 13/13 stadial compliance.
- **Stateful IPC surface** (Sprint 54): Path A session methods for multi-request tensor workflows.
- **Precision route advisory** (Sprint 58): Method advising f32 vs f64 vs CPU route based on hardware capability matrix.

---

## Neural & Machine Learning — training and inference on GPU

ML capabilities absorbed from hotSpring, neuralSpring, and groundSpring into unified IPC methods with zero-copy buffer reuse.

- **MLP training** (Wave 73→Sprint 55): `SimpleMlp::from_dims()` Xavier init; dimension-shorthand `ml.mlp_train` wire contract; batch-256 perf <1s.
- **LSTM zero-copy** (Wave 120): `forward_into` + `GateBuffers` pre-allocated scratch; `forward_sequence` reuses buffers across timesteps.
- **ESN predict** (Wave 53): Full E2E with trained ESN + state injection coverage tests.
- **Perceptron wire contract** (Wave 73): Matches `NEURAL_API_PERCEPTRON_DESIGN.md` — layers shorthand auto-creates and trains from scratch.
- **Nautilus sessions** (Sprint 55): Stateful training session surface for multi-epoch GPU workloads.

---

## Mathematical & Statistical Computing — cross-spring pattern absorption

Math evolved from monolithic 1046L `math.rs` to domain-decomposed modules absorbing patterns from airSpring, groundSpring, and healthSpring.

- **Module decomposition** (Wave deep debt): `math.rs` (1046L) → `math.rs` (305L) + `stats.rs` (576L) + `signal.rs` (151L).
- **Cross-spring absorption** (Sprint 73): `stats.fit_quadratic/exponential/logarithmic`, `stats.rarefaction_curve`, `stats.gamma_fit/cdf`, `signal.detect_peaks`.
- **Lattice ops** (Sprint 66): TensorSession lattice operations for groundSpring QCD pipeline.
- **Batched tridiagonal eigh** (Wave 124): `linalg.batched_tridiag_eigh` via QL with Wilkinson shifts — groundSpring Exp 012 absorption.
- **FMA evolution** (Sprint 20→Wave 142b): 238 `suboptimal_flops` → `mul_add`; function pointers replace `Box<dyn Fn>` in CPU binary ops.

---

## Anderson Transport & Membrane Physics — selective permeability

Anderson localization framework emerged as first-class transport physics with membrane permeability tensor and selectivity metrics.

- **Anderson transport module** (Wave 167): `membrane_permeability()`, `membrane_profile()`, `dimensional_localization_length()` — 14/14 tests.
- **Permeability tensor** (Paper 43): Mode-dependent transmission P(ω, W_ω, d_ω); selectivity S = max(P) - min(P) measures membrane health.
- **riboCipher genetics-layer** (Wave 114): Prefix stripping `0xEC`/`0xED`/`0xEE` on all IPC accept loops — clear/mito/nuclear tier mapping.

---

## Multi-GPU & Resource Management — pool, migration, quotas

Multi-GPU evolved from single-device assumptions to quota-aware OOM auto-migration across device pools.

- **MultiDevicePool** (Wave 119→124): `execute_with_migration()` with device exclusion and configurable retry; OOM-flagged devices skipped.
- **Quota-aware migration** (Wave 124): `execute_with_migration_quota()` carries `ResourceQuota` through migration loop.
- **Sovereign dispatch** (Wave 120→Sprint 71): `dispatch_via_sovereign()` routes tensor IPC through coralReef peer when local wgpu unavailable.
- **Cross-gate dispatch** (Wave 67): `compute.dispatch.capabilities/submit/result` — wire-compatible with hotSpring pipeline contract.
- **ToadStool binary routing** (Wave 67 FRAGO): Mode 1 shader binary resolves compute provider via `ipc.resolve`; falls back with `"routed": false` transparency.

---

## coralReef Integration — sovereign GPU compilation peer

barraCuda and coralReef evolved from ad-hoc IPC to structured shader compilation delegation with provenance tracking.

- **shader.compile.wgsl IPC** (Wave 123): coralReef operational via TCP JSON-RPC; 27ms WGSL→PTX; BLAKE3 provenance on ironGate.
- **Sovereign-dispatch feature** (Wave 120): `barracuda-core` feature passthrough; `dispatch_capabilities` reports `gpu.sovereign_ipc`.
- **Dual-target depot** (Wave 120): Upstream impulse for `x86_64-unknown-linux-gnu` GPU builds alongside musl static.
- **PrecisionBrain integration** (Sprint 22f): Dispatch metadata and coralReef CPU compilation path for headless validation.

---

## Transport, BTSP & Session Security — wire stack evolution

Transport matured from monolithic 805L file to modular server/connection/dispatch with BTSP Phase 3 and cross-architecture support.

- **Transport module split** (Wave 67→156p): `transport.rs` (780L) → `transport/` (server.rs + connection.rs + dispatch.rs + transport_config.rs).
- **BTSP Phase 3** (Sprint 51→Wave 151b): `btsp.negotiate` with ChaCha20-Poly1305; client-side ClientHello for bearDog strict mode.
- **Phase 2 transport abstraction** (Wave 143b): `TransportListener` enum unifies UDS + TCP server-side bind/accept; 5 `#[cfg(unix)]` gates removed.
- **Bind mode CLI** (Wave 109): `--bind-mode` / `PRIMAL_BIND_MODE` for launcher-controlled socket/TCP selection.
- **Cross-architecture** (Wave 141a): Full `x86_64-pc-windows-gnu` cross-compile; platform-agnostic transport dispatch.
- **Hardcoding → capability-based** (Wave 141a→151b): Socket prefixes, BTSP provider discovery prefer `BTSP_PROVIDER_SOCKET` over legacy `BEARDOG_*`.

---

## Architecture & Engineering Discipline — reference-quality codebase

barraCuda achieved 12-axis deep debt audit compliance through systematic refactoring and dependency purity.

- **12-axis audit** (Wave 129→156p): Zero files >800L, zero unsafe in production, zero TODO/FIXME/HACK, zero `Result<T,String>`, zero production mocks.
- **MSRV 1.92 + Edition 2024** (Wave 128): let-chain modernization; `#[expect]` over `#[allow]`.
- **bincode → postcard** (Wave 128): RUSTSEC-2025-0141 resolved; format tag 2.
- **LatencyModel enum dispatch** (Wave 141a): `trait → enum` eliminates `Box<dyn LatencyModel>` heap allocation per shader compile.
- **ODE integrator generics** (Wave 142b): `&impl Fn` replaces `Box<dyn Fn>` — unboxed closure parameters.
- **Test promotion** (Wave 156k): 13 double-gated `#[ignore]` tests promoted to active — all pass on real GPU and llvmpipe.
- **Quality gates** (Wave 156p): 4,624 tests; fmt, clippy, doc, deny, check all green.

---

## Health, Discovery & Deployment — operational readiness

Operational surfaces evolved from always-healthy stubs to real probes with deployment behavior convergence across the ecosystem.

- **health.version RPC** (Sprint 69): Version metadata on health surface for tower atomic validation and depot compatibility checks.
- **Deployment behavior convergence** (Wave 47): `--socket` visible alias for `--unix` per DEPLOYMENT_BEHAVIOR_STANDARD — uniform launcher invocation.
- **GNU depot validated** (Wave 128): E2E fetch from golgi depot, BLAKE3 checksum verified, glibc binary runs on ironGate RTX 5070.
- **Ecosystem namespace constants** (Wave 123): `env_keys::DEFAULT_ECOSYSTEM_SOCKET_NAMESPACE` — single source of truth for socket prefix resolution.
- **Wave 22 stadial gate** (Sprint 70): Stadial compliance checklist cleared for primalSpring composition audits.

---

*Full date-sorted history: `CHANGELOG_ARCHIVE.md`*
*Last compressed: Wave 172*
