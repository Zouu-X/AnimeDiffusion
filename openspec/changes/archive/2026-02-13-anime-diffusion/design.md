## Context

This change introduces a production-style data generation pipeline that creates 100,000 anime face images at 512x512 using `hakurei/waifu-diffusion-v1-4` (loaded via `StableDiffusionPipeline.from_single_file()` from the `wd-1-4-anime_e1.ckpt` checkpoint) and packages outputs in WebDataset shards. The current repository has no existing OpenSpec capability specs, so this design must define a clear architecture that can be implemented in modular steps and resumed after interruption. The primary constraints are deterministic reproducibility, large-volume output handling, and strict output conformance (image size/count and shard integrity).

## Goals / Non-Goals

**Goals:**
- Generate exactly 100,000 images with fixed 512x512 resolution.
- Use Waifu Diffusion v1.4 as the single generation model baseline.
- Emit WebDataset-compatible shards (`.tar`) with per-sample image and metadata payloads.
- Ensure deterministic behavior via seed management and persisted progress state.
- Support restart/resume without duplicating already materialized samples.
- Provide automated validation checks before marking the run complete.

**Non-Goals:**
- Training or fine-tuning any diffusion model.
- Building a generic multi-model generation platform in this change.
- Curating prompt quality through human review loops.
- Defining downstream model-training pipelines that consume the dataset.

## Decisions

1. Pipeline structure: staged, file-backed workflow
- Decision: Split the pipeline into discrete stages: `plan -> generate -> package -> validate`.
- Rationale: Stage boundaries simplify retries, observability, and partial reruns.
- Alternatives considered:
  - Single monolithic script: simpler short-term, but fragile for long-running 100k generation jobs.
  - Fully distributed workflow engine: scalable, but unnecessary complexity for initial version.

2. Runtime stack: Diffusers-based inference with explicit scheduler/precision controls
- Decision: Use Hugging Face Diffusers pipeline loading `hakurei/waifu-diffusion-v1-4` via `from_single_file()`, with configuration captured in run manifests.
- Rationale: Native compatibility with the requested model and predictable configuration surface.
- Alternatives considered:
  - Custom latent diffusion inference code: more control, much higher implementation risk.
  - External hosted inference APIs: less local complexity but weaker reproducibility control and potential cost/availability risk.

3. Reproducibility model: deterministic sample manifest + global sample index
- Decision: Precompute sample definitions (`sample_id`, prompt, seed, cfg/steps if configurable) and persist to manifest files.
- Rationale: Decouples sampling plan from execution order; resume can skip completed sample IDs safely.
- Alternatives considered:
  - On-the-fly random prompt/seed generation: lower upfront work but poor reproducibility and restart behavior.

4. Intermediate output format: write PNG + JSON sidecars before sharding
- Decision: Materialize each sample as `sample_id.png` and `sample_id.json` in a temporary workspace, then stream into WebDataset shards.
- Rationale: Makes validation/resume and failure debugging straightforward.
- Alternatives considered:
  - Direct tar streaming during generation: lower disk usage, but harder resume and failure recovery.

5. WebDataset packaging policy: fixed-size shards with deterministic naming
- Decision: Package with shard size target (for example 1,000 samples/shard) and names like `anime-face-000000.tar`.
- Rationale: Deterministic naming aids downstream indexing and incremental re-checks.
- Alternatives considered:
  - Variable-size shards: simpler packing logic, weaker predictability for consumers.

6. Validation gates: hard completion criteria
- Decision: Completion requires all gates to pass:
  - count == 100,000 samples
  - all images decode and are exactly 512x512
  - each sample has required metadata fields
  - shard index coverage has no gaps/duplicates
- Rationale: Prevent silent data quality regressions and enforce contract for downstream use.
- Alternatives considered:
  - Soft warnings only: faster completion at cost of unreliable datasets.

## Risks / Trade-offs

- [Long runtime / GPU instability] -> Mitigation: checkpoint progress per batch, idempotent sample IDs, resume from manifest state.
- [Disk pressure from temporary PNG+JSON artifacts] -> Mitigation: configurable cleanup mode after successful shard packaging; monitor free space before start.
- [Prompt distribution bias] -> Mitigation: document prompt generation strategy and expose pluggable prompt source for future iteration.
- [Library/version drift affecting reproducibility] -> Mitigation: persist model revision, dependency versions, and core inference parameters in run manifest.
- [Corrupted partial shards after interruption] -> Mitigation: write shards atomically to temp name then rename on finalize; revalidate on resume.

## Migration Plan

1. Add pipeline module skeleton and shared config schema.
2. Implement manifest planner and deterministic sample ID/seed generation.
3. Implement generation runner with batch checkpoints.
4. Implement WebDataset packager with deterministic shard names.
5. Implement validator and final run summary report.
6. Run a small-scale dry run (for example 1,000 samples) to verify behavior.
7. Execute full 100,000-sample run.
8. Publish completion report and dataset manifest for downstream consumers.

Rollback strategy:
- If generation/packaging fails, keep prior completed artifacts untouched and resume from last successful checkpoint.
- If validation fails after full run, mark dataset as incomplete and rerun only failing sample/shard segments.

## Open Questions

- Which prompt source should be used for initial corpus diversity (static template set, file-based prompt list, or generated permutations)?
- What default inference parameters should be locked in spec (steps, guidance scale, negative prompt baseline)?
- What is the target shard size for optimal downstream training throughput in this environment?
- Should we include optional caption text fields beyond generation parameters in metadata for future multimodal use?
## Answer for the open questions

- Check the codebase, I have created a prompt jsonl file for diffusion model use. file path: ./output/full/prompts.jsonl
- Default inference parameters: steps=20, guidance_scale=7.5, Sampler=EulerAncestralDiscreteScheduler, Seed: Set unique seed for each sample, Resolution=512x512
- Shard size: 1000 samples per shard.
- No