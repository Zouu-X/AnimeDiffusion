## Context

`image_gen` inference currently runs on Databricks A10G with PyTorch `2.3.1+cu121` and Diffusers `0.36.0`. The current xFormers attention path is not working in this environment, preventing expected GPU attention acceleration and adding uncertainty when debugging performance regressions.

This change keeps model behavior aligned with existing generation requirements while replacing the acceleration mechanism with a backend-native path supported by the current runtime stack.

## Goals / Non-Goals

**Goals:**
- Replace xFormers attention acceleration with PyTorch SDPA in the inference path.
- Enforce fused SDPA kernels and disable math fallback so acceleration failures are explicit.
- Apply `torch.compile` only to UNet to improve runtime throughput without broad compile risk.
- Keep inference on GPU only (no CPU offload).
- Add observability checks/logging that make acceleration state visible during startup/run.

**Non-Goals:**
- Changing model baseline, scheduler defaults, seeds, output resolution, or dataset semantics.
- Compiling full pipeline components beyond UNet.
- Supporting CPU-offloaded execution in this optimization path.
- Introducing new acceleration dependencies (for example, reworking xFormers support).

## Decisions

1. **Use SDPA as the attention backend instead of xFormers**
   - Decision: Configure attention to use PyTorch SDPA-compatible processors for the Diffusers pipeline.
   - Rationale: SDPA is native to PyTorch and better aligned with the current Databricks runtime than the currently broken xFormers path.
   - Alternative considered: Repair and keep xFormers. Rejected for now because immediate goal is stable acceleration in the current environment with lower integration risk.

2. **Force fused-kernel-only SDPA mode for debugging clarity**
   - Decision: Enable fused SDPA kernels and disable math fallback.
   - Rationale: Silent fallback can hide misconfiguration and skew performance analysis. Fused-only mode causes acceleration issues to surface quickly.
   - Alternative considered: Allow fallback for resilience. Rejected because this change prioritizes explicit acceleration verification over graceful degradation.

3. **Compile only UNet with `torch.compile`**
   - Decision: Restrict compilation to UNet and keep other pipeline modules eager.
   - Rationale: UNet dominates inference compute; limiting compile scope captures most benefit while reducing graph-break/debug complexity.
   - Alternative considered: Compile entire pipeline. Rejected due to higher startup overhead, broader failure surface, and reduced debuggability.

4. **No CPU offload in this path**
   - Decision: Keep execution fully on GPU and avoid CPU offload mechanisms.
   - Rationale: CPU offload introduces transfer overhead and confounds acceleration diagnostics.
   - Alternative considered: Optional offload for memory headroom. Rejected because target hardware (A10G) and goal (speed/debug clarity) favor GPU-resident execution.

5. **Add explicit runtime verification/logging of acceleration state**
   - Decision: Emit startup/runtime signals for SDPA mode and compile enablement; fail fast or surface clear errors when fused path is unavailable.
   - Rationale: Prevents ambiguous performance outcomes and shortens debugging loops.
   - Alternative considered: Minimal logging only. Rejected because this change is performance-focused and needs hard evidence of active acceleration.

## Risks / Trade-offs

- **[Risk] Fused kernels may be unavailable for specific shapes/dtypes/runtime combinations** -> **Mitigation:** Validate on startup with representative inference settings and provide explicit error messages to guide fallback troubleshooting.
- **[Risk] `torch.compile` can increase cold-start latency** -> **Mitigation:** Scope compile to UNet only and measure warm-up separately from steady-state throughput.
- **[Risk] Fused-only policy may reduce robustness versus fallback-enabled mode** -> **Mitigation:** Keep this behavior behind inference configuration for controlled rollout and diagnostics.
- **[Risk] Diffusers attention API differences across versions** -> **Mitigation:** Implement against `0.36.0` interfaces and add version-guarded checks in the configuration path.

## Migration Plan

1. Introduce SDPA-based attention configuration and remove/disable xFormers usage in the inference path.
2. Add SDPA kernel policy setup (fused enabled, math fallback disabled) at pipeline initialization.
3. Apply `torch.compile` to UNet only after model load and before serving inference requests.
4. Add runtime diagnostics/assertions that report active acceleration configuration.
5. Validate in Databricks A10G with fixed prompts and compare baseline vs optimized latency/throughput.
6. Roll out behind config gating; if instability appears, disable the optimization gate while preserving previous functional path.

## Open Questions

- Should fused-only enforcement fail hard at startup, or permit an explicit opt-out flag for emergency runs?
- Which compile mode/settings (`mode`, `dynamic`, `fullgraph`) provide best stability/performance for this workload?
- Do we need a dedicated benchmark harness artifact for regression tracking, or are existing run logs sufficient?

## Answers for Open Questions

- Fail hard at startup
- mode="reduce-overhead", dynamic = False, fullgraph = False
- Exising run logs sufficient
