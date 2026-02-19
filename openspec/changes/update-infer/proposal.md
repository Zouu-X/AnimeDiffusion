## Why

Current image generation inference on Databricks A10G (PyTorch 2.3.1+cu121, Diffusers 0.36.0) is not benefiting from attention acceleration because the current xFormers path is not functional in this environment. This creates unnecessary latency and makes performance troubleshooting difficult.

## What Changes

- Replace xFormers-based attention acceleration with PyTorch SDPA for inference.
- Force SDPA to use fused attention kernels and disable math fallback to make acceleration status explicit and debuggable.
- Add `torch.compile` to inference, limited to the UNet module only.
- Keep the rest of the pipeline uncompiled and keep execution fully on GPU (no CPU offload).
- Add runtime validation/logging so failed fused-kernel activation is surfaced immediately during debugging.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `waifu-diffusion-inference-config`: Update inference backend/runtime requirements to use SDPA fused-kernel mode, disable math fallback, compile UNet only, and disallow CPU offload for this path.

## Impact

- Affected code:
  - Inference pipeline setup and attention processor configuration.
  - Runtime torch backend flags/settings for SDPA kernel selection.
  - UNet initialization path to apply `torch.compile`.
- Operational impact:
  - Improved inference throughput/latency on Databricks A10G when fused SDPA is active.
  - Clearer failure mode when acceleration is not enabled (no silent math fallback).
- Dependencies/environment:
  - Behavior is scoped to PyTorch 2.3.1+cu121 and Diffusers 0.36.0 runtime assumptions.
