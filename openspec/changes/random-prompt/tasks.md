## 1. Generator Foundation

- [ ] 1.1 Create the random prompt generator module entrypoint and wire deterministic base-seed configuration.
- [ ] 1.2 Define typed prompt schema components (face shape, eyes, nose, mouth, hair, expression, pose, style modifiers).
- [ ] 1.3 Implement shard-aware generation (deterministic shard seeds and stable record ordering).
- [ ] 1.4 Implement output record model with required fields (`id`, `positive_prompt`, `negative_prompt`, `shard_id`, `seed`).

## 2. Vocabulary and Sampling Controls

- [ ] 2.1 Add configurable vocab/weight sources for each prompt component group.
- [ ] 2.2 Implement weighted samplers for each attribute group and integrate compatibility checks between components.
- [ ] 2.3 Enforce age-style restrictions so only `teen` and `adult` buckets are sampled.
- [ ] 2.4 Enforce frontal-pose constraints so sampled yaw/pitch remain within `[-15, 15]` degrees.

## 3. Quality Constraints and Validation

- [ ] 3.1 Implement dual-channel prompt assembly (positive + non-empty negative prompt for every record).
- [ ] 3.2 Add lint rules to reject malformed, contradictory, or banned-quality prompt compositions.
- [ ] 3.3 Implement rejection-and-regenerate flow so invalid candidates are replaced until valid output is produced.
- [ ] 3.4 Add rejection telemetry counters grouped by reason (lint, dedup, diversity backpressure).

## 4. Diversity and Deduplication

- [ ] 4.1 Implement corpus-level diversity tracking for eyes, hair, expression, and pose variant counts.
- [ ] 4.2 Enforce minimum thresholds (eyes>=20, hair>=20, expression>=8, pose>=8) before final publish.
- [ ] 4.3 Implement normalized semantic near-duplicate detection and replacement logic.
- [ ] 4.4 Export machine-readable diversity reports and cross-feature summaries per run.

## 5. Model Compatibility and Output Pipeline

- [ ] 5.1 Add Waifu Diffusion v1.4 compatibility mode and bind it as the default target for this change.
- [ ] 5.2 Implement final prompt export pipeline for full-corpus output and run metadata manifests.
- [ ] 5.3 Add corpus finalization guard that blocks publishing when diversity thresholds are not met.
- [ ] 5.4 Add CLI/config switches for pilot runs (for example 1K) and full runs (100K).

## 6. Verification and Acceptance

- [ ] 6.1 Add deterministic rerun test verifying identical outputs under identical config and seed.
- [ ] 6.2 Add rule tests for age restriction, frontal pose limits, and required output fields.
- [ ] 6.3 Add diversity-gate tests verifying publish is blocked when thresholds are unmet.
- [ ] 6.4 Execute pilot generation, calibrate weights/constraints
- [ ] 6.5 Write a README.md for this change
