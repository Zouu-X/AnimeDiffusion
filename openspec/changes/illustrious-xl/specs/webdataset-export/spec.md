## MODIFIED Requirements

### Requirement: Package outputs as deterministic WebDataset shards
The system SHALL package finalized samples into WebDataset `.tar` shards using `webdataset.TarWriter`, with deterministic shard names and ordering. Each sample SHALL be written with `__key__` set to the sample ID and extensions `.png` and `.json` for image and metadata respectively.

#### Scenario: Shard naming is deterministic
- **WHEN** shard files are written for a completed batch range
- **THEN** each shard SHALL follow a stable zero-padded naming scheme (`anime-face-{shard_id:06d}.tar`) and deterministic sample ordering

#### Scenario: Shards are written using webdataset library
- **WHEN** a shard is created
- **THEN** it SHALL be written using `webdataset.TarWriter` with atomic write (temp file + rename)

### Requirement: Export required per-sample metadata
The system SHALL include per-sample metadata fields for prompt, negative_prompt, seed, model_id, resolution, and inference parameters (num_inference_steps, guidance_scale, scheduler) in exported JSON payloads.

#### Scenario: Metadata fields are present in shard payloads
- **WHEN** a sample is packaged into a shard
- **THEN** its metadata JSON SHALL include prompt, negative_prompt, seed, model_id, resolution, num_inference_steps, guidance_scale, and scheduler

## ADDED Requirements

### Requirement: Declare webdataset as a project dependency
The project SHALL declare `webdataset>=0.2.100` in `pyproject.toml` dependencies.

#### Scenario: Dependency is installable
- **WHEN** `pip install -e .` is executed
- **THEN** the `webdataset` package SHALL be installed at version 0.2.100 or higher
