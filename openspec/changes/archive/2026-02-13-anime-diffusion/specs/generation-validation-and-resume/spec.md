## ADDED Requirements

### Requirement: Validate completion gates before marking run complete
The system SHALL mark a run complete only when sample count, image size, metadata presence, and shard integrity gates all pass.

#### Scenario: Completion is blocked on failed gate
- **WHEN** any validation gate fails during finalization
- **THEN** the run SHALL remain incomplete and SHALL report failing gate details

### Requirement: Detect gaps and duplicates across sample IDs
The system SHALL validate that finalized sample IDs cover the expected range with no gaps and no duplicates.

#### Scenario: ID coverage validation passes
- **WHEN** validation scans all finalized samples
- **THEN** it SHALL confirm exactly one occurrence of each expected sample ID

### Requirement: Resume safely from persisted progress
The system SHALL support interruption recovery by resuming from persisted progress state and skipping already finalized samples.

#### Scenario: Resume avoids duplicate regeneration
- **WHEN** a previously interrupted run is resumed
- **THEN** the pipeline SHALL not regenerate samples already marked as finalized in progress state

