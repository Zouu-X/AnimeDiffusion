## MODIFIED Requirements

### Requirement: Dual-channel prompts with quality guards
The system SHALL emit a positive prompt and a negative prompt for every record, and SHALL apply lint rules that reject malformed, contradictory, or banned-quality prompt compositions before export. The looking-direction contradiction lint check SHALL be removed since no directional pose tokens appear in prompts after the frontal-pose change.

#### Scenario: Invalid prompt is rejected
- **WHEN** a generated prompt violates a lint rule
- **THEN** the system SHALL reject the prompt record and regenerate a replacement

#### Scenario: Every record has negative constraints
- **WHEN** a prompt record is accepted
- **THEN** the record SHALL include a non-empty negative prompt string

#### Scenario: Anti-side-view negatives always included
- **WHEN** a negative prompt is assembled
- **THEN** the negative prompt SHALL include all tokens from the `anti_side_view` conditional negative group

#### Scenario: Extreme-closeup negatives always included
- **WHEN** a negative prompt is assembled
- **THEN** the negative prompt SHALL include all tokens from the `extreme_closeup` conditional negative group

#### Scenario: Looking-direction contradiction check removed
- **WHEN** lint rules are applied to a prompt record
- **THEN** no lint rule SHALL check for contradictions between closed-eyes and looking-direction tokens

### Requirement: Enforce minimum diversity coverage
The system SHALL enforce minimum diversity coverage across the full 100,000-prompt corpus with thresholds of at least 20 eye variants, 20 hair variants, 8 expression variants, and 5 accessories variants.

#### Scenario: Diversity thresholds pass
- **WHEN** corpus generation is finalized
- **THEN** aggregate diversity metrics SHALL meet or exceed eyes=20, hair=20, expression=8, and accessories=5 distinct variants

#### Scenario: Threshold failure blocks finalization
- **WHEN** any diversity threshold is not met
- **THEN** the system SHALL mark the corpus as failed and SHALL not publish it as final output
