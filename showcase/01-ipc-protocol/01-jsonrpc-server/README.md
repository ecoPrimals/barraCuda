# 01 — JSON-RPC Server

Demonstrates barraCuda running as an IPC server in the ecoPrimals ecosystem.

## What It Shows

- Starting the barraCuda IPC server (Unix socket + TCP)
- JSON-RPC 2.0 method calls: health, capabilities, device list, GPU validation
- Semantic method naming: `{domain}.{operation}` per wateringHole standard
- Dual transport: Unix domain socket (default) + TCP fallback

## Prerequisites

Build the barraCuda binary first:

```bash
cd ../../.. && cargo build --release -p barracuda-core
```

## Run

```bash
./demo.sh
```

## Methods Exercised

| Method | Purpose |
|--------|---------|
| `primal.info` | Primal identity and version |
| `primal.capabilities` | Advertised capabilities |
| `device.list` | Discovered GPU devices |
| `health.check` | Health status |
| `tolerances.get` | Tolerance thresholds |
| `validate.gpu_stack` | GPU validation canary |
