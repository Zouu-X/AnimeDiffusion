## ADDED Requirements

### Requirement: Generate 100K reproducible prompts
The system SHALL generate a corpus of exactly 100,000 prompts for downstream image synthesis, and SHALL support deterministic regeneration from the same configuration and base seed.

#### Scenario: Deterministic rerun
- **WHEN** the generator is executed twice with identical vocabulary files, weights, base seed, and shard settings
- **THEN** the produced prompt records SHALL be byte-for-byte identical after stable ordering

#### Scenario: Target count is enforced
- **WHEN** generation completes successfully
- **THEN** the exported dataset SHALL contain exactly 100,000 valid prompt records

### Requirement: Structured anime-face prompt composition
The system SHALL compose each positive prompt from structured facial feature fields including face shape, eyes, nose, mouth, hair, expression, and pose-related descriptors to keep outputs focused on anime faces.

#### Scenario: Prompt uses facial feature schema
- **WHEN** a prompt record is generated
- **THEN** its positive prompt SHALL include tokens from the defined facial-feature component groups

### Requirement: Frontal-face pose constraints
The system SHALL constrain pose descriptors so generated prompts target mostly frontal faces, with yaw and pitch each constrained to 15 degrees or less in absolute value.

#### Scenario: Pose remains frontal
- **WHEN** pose attributes are sampled for a prompt
- **THEN** sampled yaw and pitch SHALL each be within the inclusive range of -15 to 15 degrees

### Requirement: Age-style bucket restrictions
The system SHALL include only teen-coded and adult-coded appearance buckets and SHALL exclude baby/child and old/elderly buckets.

#### Scenario: Allowed age buckets only
- **WHEN** age-style attributes are sampled
- **THEN** the selected bucket SHALL be either teen or adult

### Requirement: Waifu Diffusion v1.4 prompt compatibility
The system SHALL produce prompt text compatible with Waifu Diffusion v1.4 tokenization and usage patterns.

#### Scenario: Model target is explicit
- **WHEN** prompt generation configuration is loaded
- **THEN** the model target SHALL be set to Waifu Diffusion v1.4 compatibility mode

### Requirement: Generation-ready export format
The system SHALL export each prompt record with at least a unique identifier, positive prompt, negative prompt, shard identifier, and seed trace for downstream batch synthesis and auditability.

#### Scenario: Required output fields exist
- **WHEN** a prompt record is written to output
- **THEN** the record SHALL include id, positive_prompt, negative_prompt, shard_id, and seed fields
