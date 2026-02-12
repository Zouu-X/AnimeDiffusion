## 1. Configuration and Pipeline Skeleton

- [ ] 1.1 Create a shared run configuration schema for model id, output paths, batch size, shard size, target count, and inference defaults
- [ ] 1.2 Implement CLI entrypoints for `plan`, `generate`, `package`, and `validate` stages
- [ ] 1.3 Add run-manifest writing that records model version, dependency versions, and resolved run settings

## 2. Deterministic Planning and Progress State

- [ ] 2.1 Implement a planner that produces exactly 100,000 deterministic sample IDs with shard-aligned ranges
- [ ] 2.2 Implement deterministic seed assignment per sample ID and persist seeds in the manifest
- [ ] 2.3 Add persisted progress state storage that marks finalized samples and completed shard outputs

## 3. Waifu Diffusion Inference Execution

- [ ] 3.1 Implement inference pipeline loading for `hakurei/waifu-diffusion-v1-3`
- [ ] 3.2 Apply default inference settings (`steps=20`, `guidance_scale=7.5`, `EulerAncestralDiscreteScheduler`) with override support
- [ ] 3.3 Implement batched generation of 512x512 PNG outputs plus per-sample JSON sidecars
- [ ] 3.4 Add checkpoint commits after each batch so interrupted runs can resume without regenerating finalized samples

## 4. WebDataset Packaging

- [ ] 4.1 Implement shard writer that packages PNG and JSON pairs into deterministic `.tar` shard files
- [ ] 4.2 Enforce shard sizing policy of 1,000 samples per non-final shard and a remainder final shard
- [ ] 4.3 Add deterministic shard naming (e.g., `anime-face-000000.tar`) and atomic temp-to-final rename on completion

## 5. Validation and Completion Gates

- [ ] 5.1 Implement sample count validation to require exactly 100,000 unique finalized sample IDs
- [ ] 5.2 Implement image integrity validation to decode images and verify exact 512x512 dimensions
- [ ] 5.3 Implement metadata validation to require prompt, seed, model identifier, and inference parameter fields per sample
- [ ] 5.4 Implement coverage validation for shard/sample indexes to detect gaps and duplicates
- [ ] 5.5 Block completion status when any validation fails and emit structured failure reasons

## 6. Resume, Reporting, and Dry Runs

- [ ] 6.1 Implement resume logic that skips samples already marked finalized in progress state
- [ ] 6.2 Add a run summary report with throughput, shard counts, validation results, and failure breakdowns
- [ ] 6.3 Execute a 1,000-sample dry run and verify deterministic plan, packaging layout, and validation outputs
- [ ] 6.4 Run the full 100,000-sample workflow and publish the final manifest/report artifacts
