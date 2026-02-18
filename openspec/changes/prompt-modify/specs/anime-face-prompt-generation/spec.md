## MODIFIED Requirements

### Requirement: Structured anime-face prompt composition
The system SHALL compose each positive prompt from structured facial feature fields including face shape, eyes, nose, mouth, hair, expression, optional accessories, and constant frontal pose tags to keep outputs focused on anime faces.

#### Scenario: Prompt uses facial feature schema
- **WHEN** a prompt record is generated
- **THEN** its positive prompt SHALL include tokens from the defined facial-feature component groups

#### Scenario: Accessories token positioning
- **WHEN** a prompt record includes a non-empty accessories value
- **THEN** the accessories token SHALL appear after the expression token and before the constant frontal pose tags

### Requirement: Frontal-face pose constraints
The system SHALL include the constant tags "frontal face" and "looking at viewer" in every positive prompt, replacing the previous yaw/pitch numeric sampling system.

#### Scenario: Constant frontal tags present
- **WHEN** a prompt record is generated
- **THEN** the positive prompt SHALL contain the literal tokens "frontal face" and "looking at viewer"

#### Scenario: No numeric pose sampling
- **WHEN** a prompt record is generated
- **THEN** no yaw or pitch numeric values SHALL be sampled or converted to directional text tags

## ADDED Requirements

### Requirement: Accessory composition
The system SHALL support an optional `accessories` field (`str = ""`) on each prompt, sampled from a weighted vocab category that includes a no-accessory empty-string entry. When non-empty, the accessory token SHALL be positioned after expression and before frontal pose tags.

#### Scenario: Accessory sampled and included
- **WHEN** the weighted sampler selects a non-empty accessories value
- **THEN** the positive prompt SHALL include the accessory token after expression and before the constant frontal tags

#### Scenario: No-accessory outcome
- **WHEN** the weighted sampler selects the empty-string accessories entry
- **THEN** the positive prompt SHALL NOT include any accessory token

#### Scenario: Vocab includes no-accessory entry
- **WHEN** the accessories vocab category is loaded
- **THEN** it SHALL contain an entry with value `""` and a positive weight
