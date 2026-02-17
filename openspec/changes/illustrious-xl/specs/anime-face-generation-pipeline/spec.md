## MODIFIED Requirements

### Requirement: Enforce fixed output image resolution
The system SHALL produce generated images at exactly 1024x1024 pixels for all finalized samples.

#### Scenario: Image dimensions are fixed
- **WHEN** a sample image is materialized by the generation pipeline
- **THEN** the image width SHALL be 1024 pixels and the image height SHALL be 1024 pixels
