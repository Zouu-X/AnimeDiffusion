## Why

We need a reproducible way to generate a large anime-face image corpus for model training and evaluation without manual curation overhead. Building a dedicated pipeline now enables consistent, scalable generation of 100,000 images at 512x512 resolution using a single known model baseline.

## What Changes

- Add an end-to-end generation pipeline that uses `hakurei/waifu-diffusion-v1-3` to synthesize anime face images at fixed 512x512 resolution.
- Add configurable batch processing to produce a total of 100,000 images (0.1 million) with deterministic sharding and resumable progress tracking.
- Add dataset packaging logic to write outputs in WebDataset format (tar shard + sample key conventions).
- Add metadata capture per sample (for example prompt, seed, and generation parameters) so downstream consumers can trace provenance.
- Add validation checks to ensure expected sample count, shard integrity, and image size conformity before dataset completion is reported.

## Capabilities

### New Capabilities
- `anime-face-generation-pipeline`: Orchestrate batched text-to-image generation to produce exactly 100,000 anime face images at 512x512.
- `waifu-diffusion-inference-config`: Standardize model loading and inference parameterization for `hakurei/waifu-diffusion-v1-3`, including reproducibility controls.
- `webdataset-export`: Package generated samples and metadata into WebDataset-compliant shards with predictable naming and indexing.
- `generation-validation-and-resume`: Validate output completeness/integrity and support safe resume after interruption.

### Modified Capabilities
- None.

## Impact

- Affected code: new pipeline modules for prompt/seed scheduling, model inference, sample serialization, sharding, and validation.
- Affected dependencies/systems: Diffusers/Hugging Face model loading stack, image serialization libraries, and WebDataset writing utilities.
- Affected outputs: a new generated dataset artifact in WebDataset shard format sized for 100,000 512x512 anime face samples.
