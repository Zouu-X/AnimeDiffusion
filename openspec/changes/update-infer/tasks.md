## 1. Inference Runtime Configuration

- [x] 1.1 Locate the current `image_gen` inference initialization path and identify all xFormers-related setup points.
- [x] 1.2 Add runtime configuration for SDPA kernel policy that enables fused kernels and disables math fallback.
- [x] 1.3 Remove or bypass xFormers attention enablement so SDPA is the active attention backend.
- [x] 1.4 Ensure CPU offload is explicitly disabled for this optimization path.

## 2. SDPA Attention Backend Integration

- [x] 2.1 Configure the Diffusers pipeline attention processor path to use PyTorch SDPA-compatible attention.
- [x] 2.2 Add startup/runtime validation that verifies fused SDPA policy is active.
- [x] 2.3 Implement explicit error surfacing when fused SDPA cannot be satisfied (no silent math fallback).

## 3. UNet Compilation Optimization

- [x] 3.1 Apply `torch.compile` only to UNet after model load and before inference execution.
- [x] 3.2 Set compile options to `mode="reduce-overhead"`, `dynamic=False`, and `fullgraph=False`.
- [x] 3.3 Verify non-UNet modules (text encoder, VAE, scheduler) remain in eager mode.

## 4. Observability and Validation

- [x] 4.1 Add structured logs/metrics that report active attention backend, SDPA kernel policy, and UNet compile status at startup.
- [x] 4.2 Add a targeted runtime check/test that fails when math fallback is enabled or fused-only enforcement is not active.
- [x] 4.3 Add/update documentation comments for Databricks A10G assumptions (PyTorch 2.3.1+cu121, Diffusers 0.36.0).

## 5. Performance Verification

- [ ] 5.1 Run baseline and optimized inference measurements on Databricks A10G using fixed prompts/settings.
- [ ] 5.2 Compare latency/throughput results and record outcomes in run logs.
- [ ] 5.3 Confirm that behavior/output invariants (model baseline, scheduler defaults, seeds, resolution) remain unchanged.
