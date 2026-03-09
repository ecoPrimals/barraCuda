# 02 — toadStool Hardware Discovery

Demonstrates how toadStool's hardware inventory feeds into barraCuda's
GPU selection and precision routing.

## What It Shows

- Query toadStool for available compute hardware (GPU, NPU, CPU)
- Feed hardware capabilities into barraCuda's device selection
- Capability-based discovery: no hardcoded primal names or ports
- Graceful degradation: if toadStool is absent, barraCuda discovers locally

## The Discovery Flow

```
toadStool                    barraCuda
   |                            |
   | $XDG_RUNTIME_DIR/ecoPrimals/toadstool-core.json
   |  capabilities: ["hardware_discovery", "gpu_management"]
   |                            |
   |  toadstool.device.list     |
   |<---------------------------|
   |                            |
   |  { devices: [...] }        |
   |--------------------------->|
   |                            |
   |                   Select best GPU
   |                   Route precision
   |                   Dispatch shaders
```

## Run

```bash
# Without toadStool (barraCuda local discovery):
./demo.sh

# With toadStool running:
# (start toadStool in another terminal first)
./demo.sh
```
