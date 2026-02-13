## ADDED Requirements

### Requirement: Generate exactly 100,000 anime face images
The system SHALL orchestrate batched text-to-image generation and produce exactly 100,000 finalized samples per run.

#### Scenario: Run reaches target sample count
- **WHEN** a generation run completes without fatal errors
- **THEN** the finalized dataset SHALL contain exactly 100,000 unique sample IDs

### Requirement: Enforce fixed output image resolution
The system SHALL produce generated images at exactly 512x512 pixels for all finalized samples.

#### Scenario: Image dimensions are fixed
- **WHEN** a sample image is materialized by the generation pipeline
- **THEN** the image width SHALL be 512 pixels and the image height SHALL be 512 pixels

### Requirement: Execute deterministic batch planning
The system SHALL partition the total generation workload into deterministic batches and shard-aligned sample ranges.

#### Scenario: Batch plan is reproducible
- **WHEN** the planner is executed with the same run configuration and sample target
- **THEN** the batch boundaries and sample ID assignments SHALL be identical

