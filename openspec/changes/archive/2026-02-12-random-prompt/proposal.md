## Why

We need a scalable way to generate high-quality, diverse anime face prompts for Waifu Diffusion so we can produce a 100K-image dataset suitable for ViT training. Prompt quality and coverage directly determine dataset usefulness, while weak prompts increase failure cases and wasted generation cost.

## What Changes

- Add a random prompt generator tailored to anime face image synthesis with Waifu Diffusion.
- Define structured prompt composition focused on facial attributes (shape, eyes, nose, mouth, hair, expression, pose, accessories, lighting, style cues).
- Add quality-control prompt constraints to reduce common bad outputs (anatomical errors, artifacts, low-quality traits, off-topic composition).
- Add diversity controls to prevent mode collapse and increase coverage across facial features, style variants, and demographics-relevant visual traits.
- Add generation-ready output formatting and sampling controls to support building a 100K prompt list for downstream image generation.

## Capabilities

### New Capabilities

- `anime-face-prompt-generation`: Generate randomized, structured anime-face prompts for Waifu Diffusion with configurable distributions and reproducible sampling.
- `prompt-quality-and-diversity-constraints`: Enforce negative/avoidance constraints and diversity balancing rules so generated prompts minimize bad results while maximizing dataset variation.

### Modified Capabilities

- None.

## Impact

- Affected code: prompt construction logic, randomization/sampling module, dataset prompt export pipeline, and configuration surface for distributions/constraints.
- Affected outputs: prompt corpus used to synthesize approximately 100,000 anime face images for ViT training.
- Dependencies: Waifu Diffusion-compatible prompt syntax assumptions; optional seed management for reproducibility and auditability.
- Systems/process: data generation workflow and downstream training-data QA/validation process.
