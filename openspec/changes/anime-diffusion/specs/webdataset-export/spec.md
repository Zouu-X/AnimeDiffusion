## ADDED Requirements

### Requirement: Package outputs as deterministic WebDataset shards
The system SHALL package finalized samples into WebDataset `.tar` shards with deterministic shard names and ordering.

#### Scenario: Shard naming is deterministic
- **WHEN** shard files are written for a completed batch range
- **THEN** each shard SHALL follow a stable zero-padded naming scheme and deterministic sample ordering

### Requirement: Enforce shard size policy
The system SHALL target 1,000 samples per shard for full-size shards, except for a final remainder shard when the total is not evenly divisible.

#### Scenario: Shard sample counts follow policy
- **WHEN** all shards are created for a run
- **THEN** each non-final shard SHALL contain exactly 1,000 samples and the final shard SHALL contain the remaining samples

### Requirement: Export required per-sample metadata
The system SHALL include per-sample metadata fields for prompt, seed, model identifier, and inference parameters in exported payloads.

#### Scenario: Metadata fields are present in shard payloads
- **WHEN** a sample is packaged into a shard
- **THEN** its metadata SHALL include prompt, seed, model_id, and inference parameter values

