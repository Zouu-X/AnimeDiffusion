## Context

The change introduces a prompt-generation system that produces a large corpus of anime-face prompts for Waifu Diffusion image synthesis. The output will drive creation of a 100K-image dataset for ViT training, so generation must balance three constraints: high prompt quality, low bad-output rate, and broad diversity across facial features and presentation styles. Existing project specs are empty, so this design defines a new capability surface and an implementation path that can be validated before full-scale generation.

## Goals / Non-Goals

**Goals:**
- Generate 100,000 prompts with reproducible randomness (seeded execution).
- Keep prompt content centered on anime face composition and facial attributes.
- Reduce common failure prompts via explicit negative constraints and prompt lint rules.
- Enforce diversity via weighted sampling and distribution tracking.
- Export prompts in generation-ready format for downstream batch synthesis.

**Non-Goals:**
- Training or evaluating the ViT model itself.
- Building a full image-quality classifier in this change.
- Replacing the downstream image generation runtime.
- Solving every possible style bias; this change targets measurable diversity controls, not perfect fairness guarantees.

## Decisions

1. Structured prompt schema instead of freeform text generation
- Decision: Compose prompts from typed fields (identity cues, face shape, eyes, nose, mouth, hair, expression, angle, lighting, background minimality, style modifiers).
- Rationale: Structured composition gives controllable coverage and easier auditing compared with unconstrained text.
- Alternative considered: LLM-only freeform prompts.
- Why not alternative: Harder to bound quality and diversity, weaker reproducibility.

2. Two-channel output: positive prompt + negative prompt
- Decision: Every sample emits a paired positive prompt and negative prompt list.
- Rationale: Waifu Diffusion quality improves when common artifacts are explicitly discouraged.
- Alternative considered: Positive-only prompts.
- Why not alternative: Increased artifact rate and less robust filtering.

3. Weighted randomization with quotas and backpressure
- Decision: Use weighted categorical samplers per feature, plus global quota tracking to avoid overrepresented combinations.
- Rationale: Pure IID random sampling drifts toward dominant modes and reduces long-tail coverage.
- Alternative considered: Uniform random across all attributes.
- Why not alternative: Uniform independent sampling still causes skew in combined feature space and misses rare but important combinations.

4. Prompt linting and hard safety filters before export
- Decision: Apply rule-based validation to block malformed prompts, contradictory trait combinations, and banned quality terms.
- Rationale: Early rejection is cheaper than generating unusable images.
- Alternative considered: Rely only on downstream image inspection.
- Why not alternative: Wastes compute and delays feedback.

5. Reproducible, sharded generation pipeline
- Decision: Generate prompts in deterministic shards (e.g., 100 shards of 1,000) with per-shard seed derivation and per-shard stats.
- Rationale: Improves restartability, observability, and parallel execution safety.
- Alternative considered: Single monolithic run.
- Why not alternative: Harder recovery and higher risk of silent distribution drift.

6. Diversity observability metrics as first-class outputs
- Decision: Emit histogram/cross-tab summaries for key facial features and style attributes alongside prompt files.
- Rationale: ViT dataset utility depends on distribution coverage, not just sample count.
- Alternative considered: Count-only success metric.
- Why not alternative: 100K prompts can still be low-diversity and underperform.

## Risks / Trade-offs

- [Risk] Over-constrained prompts reduce stylistic richness. -> Mitigation: tune negative prompt rules and keep soft constraints configurable with A/B sampled slices.
- [Risk] Attribute combinatorics produce semantically awkward prompt pairs. -> Mitigation: add compatibility matrices and lint-time conflict checks.
- [Risk] Diversity controls may deviate from intended Waifu Diffusion token priors and reduce fidelity. -> Mitigation: run pilot generation on a small sample and calibrate weights before full run.
- [Risk] 100K generation artifacts may include near-duplicate prompts. -> Mitigation: canonicalize prompts and apply dedup hashes before export.
- [Risk] Reproducibility breaks across runtime versions. -> Mitigation: pin generator version/config snapshot and store seed + shard metadata.

## Migration Plan

1. Implement prompt schema, samplers, and validation modules behind a new generator entrypoint.
2. Add configuration files for attribute vocabularies, weights, and negative prompts.
3. Run a pilot generation (e.g., 5K prompts) and inspect diversity and lint rejection metrics.
4. Adjust weights/constraints based on pilot findings.
5. Generate full 100K prompts in deterministic shards and export manifests.
6. Freeze the final config + seed set used for dataset generation and hand off to image synthesis stage.
7. Rollback strategy: if quality/diversity checks fail, revert to previous config snapshot and regenerate affected shards only.

## Open Questions

- Which exact Waifu Diffusion checkpoint/version and tokenizer behavior will be treated as baseline for prompt compatibility?
- What are minimum acceptable diversity thresholds per feature family (eyes, hair, expression, pose)?
- Should we include explicit age-style buckets (e.g., teen/adult-coded appearance) as controlled attributes, and what policy constraints apply?
- How strict should deduplication be (exact string only vs normalized semantic near-dup)?
## Answers for open questions:
- Waifu Diffusion v1.4. Check here for details: "https://huggingface.co/hakurei/waifu-diffusion-v1-4"
- MinDiversityThreshold: Eyes:20, Hair:20, Expression:8, Pose:8 (Yaw and Pitch should be constrained below 15 degrees since I want to have frontal faces mainly)
- Yes, Include Teen and Adult. No baby kids or old people
- Normalized semantic near-duplication