## ADDED Requirements

### Requirement: Use Waifu Diffusion v1.3 as the generation baseline
The system SHALL load `hakurei/waifu-diffusion-v1-3` as the default and required model baseline for this change.

#### Scenario: Baseline model is selected
- **WHEN** the inference pipeline initializes for a run
- **THEN** model loading SHALL target `hakurei/waifu-diffusion-v1-3`

### Requirement: Lock default inference parameters
The system SHALL default to `steps=20`, `guidance_scale=7.5`, and `EulerAncestralDiscreteScheduler` unless explicitly overridden by run configuration.

#### Scenario: Defaults are applied
- **WHEN** no per-run overrides are provided for core inference parameters
- **THEN** inference SHALL execute using steps=20, guidance_scale=7.5, and EulerAncestralDiscreteScheduler

### Requirement: Assign deterministic per-sample seeds
The system SHALL assign and persist a unique deterministic seed for every planned sample ID before generation begins.

#### Scenario: Seed is stable per sample ID
- **WHEN** the same sample manifest is regenerated from identical inputs
- **THEN** each sample ID SHALL receive the same seed value as previous runs

