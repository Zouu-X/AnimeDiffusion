## ADDED Requirements

### Requirement: Dual-channel prompts with quality guards
The system SHALL emit a positive prompt and a negative prompt for every record, and SHALL apply lint rules that reject malformed, contradictory, or banned-quality prompt compositions before export.

#### Scenario: Invalid prompt is rejected
- **WHEN** a generated prompt violates a lint rule
- **THEN** the system SHALL reject the prompt record and regenerate a replacement

#### Scenario: Every record has negative constraints
- **WHEN** a prompt record is accepted
- **THEN** the record SHALL include a non-empty negative prompt string

### Requirement: Enforce minimum diversity coverage
The system SHALL enforce minimum diversity coverage across the full 100,000-prompt corpus with thresholds of at least 20 eye variants, 20 hair variants, 8 expression variants, and 8 pose variants.

#### Scenario: Diversity thresholds pass
- **WHEN** corpus generation is finalized
- **THEN** aggregate diversity metrics SHALL meet or exceed eyes=20, hair=20, expression=8, and pose=8 distinct variants

#### Scenario: Threshold failure blocks finalization
- **WHEN** any diversity threshold is not met
- **THEN** the system SHALL mark the corpus as failed and SHALL not publish it as final output

### Requirement: Distribution reporting for audit
The system SHALL emit machine-readable diversity reports covering per-feature counts and cross-feature summaries used to audit coverage before image synthesis.

#### Scenario: Diversity report is produced
- **WHEN** a generation run completes
- **THEN** the system SHALL export a report artifact containing per-feature distinct counts and aggregate distribution summaries

### Requirement: Semantic near-duplicate prevention
The system SHALL prevent normalized semantic near-duplicate prompts from appearing in the finalized corpus.

#### Scenario: Near-duplicate is filtered
- **WHEN** a candidate prompt is detected as a normalized semantic near-duplicate of an existing accepted prompt
- **THEN** the candidate SHALL be rejected and replaced with a newly generated prompt

### Requirement: Quality failure accounting
The system SHALL track rejection reasons and counts for lint failures, dedup failures, and diversity backpressure decisions.

#### Scenario: Rejection telemetry exists
- **WHEN** generation run metadata is exported
- **THEN** it SHALL include counts grouped by rejection reason category
