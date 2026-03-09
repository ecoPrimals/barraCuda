# 01 — JSON-RPC Server

Demonstrates barraCuda running as an IPC server in the ecoPrimals ecosystem.

## What It Shows

- Starting the barraCuda IPC server (Unix socket + TCP)
- JSON-RPC 2.0 method calls: health, capabilities, device list, GPU validation
- Semantic method naming: `barracuda.{domain}.{operation}`
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
| `barracuda.primal.info` | Primal identity and version |
| `barracuda.primal.capabilities` | Advertised capabilities |
| `barracuda.device.list` | Discovered GPU devices |
| `barracuda.health.check` | Health status |
| `barracuda.tolerances.get` | Tolerance thresholds |
| `barracuda.validate.gpu_stack` | GPU validation canary |
