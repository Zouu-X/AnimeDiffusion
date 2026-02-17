## MODIFIED Requirements

### Requirement: Use Waifu Diffusion v1.3 as the generation baseline
The system SHALL load Illustrious XL v2 from a local `.safetensors` checkpoint as the default and required model baseline, using the `StableDiffusionXLPipeline` from diffusers.

#### Scenario: Baseline model is selected
- **WHEN** the inference pipeline initializes for a run
- **THEN** model loading SHALL use `StableDiffusionXLPipeline.from_single_file()` with the configured checkpoint path

#### Scenario: SDXL pipeline is used
- **WHEN** the model checkpoint is loaded
- **THEN** the pipeline SHALL be an instance of `StableDiffusionXLPipeline`

### Requirement: Lock default inference parameters
The system SHALL default to `steps=28`, `guidance_scale=7.0`, and `EulerAncestralDiscreteScheduler` unless explicitly overridden by run configuration.

#### Scenario: Defaults are applied
- **WHEN** no per-run overrides are provided for core inference parameters
- **THEN** inference SHALL execute using steps=28, guidance_scale=7.0, and EulerAncestralDiscreteScheduler
