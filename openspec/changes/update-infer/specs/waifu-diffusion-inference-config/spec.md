## ADDED Requirements

### Requirement: Use PyTorch SDPA attention backend for inference
The inference system SHALL use PyTorch Scaled Dot-Product Attention (SDPA) as the attention acceleration backend and SHALL NOT depend on xFormers for this path.

#### Scenario: SDPA backend is configured
- **WHEN** the generation pipeline is initialized for inference
- **THEN** attention processing SHALL be configured to execute through PyTorch SDPA-compatible attention processors

#### Scenario: xFormers path is not used
- **WHEN** inference runtime configuration is applied
- **THEN** the system SHALL disable or bypass xFormers-specific attention configuration

### Requirement: Enforce fused-kernel SDPA and disable math fallback
The inference system SHALL force SDPA execution to fused GPU kernels and SHALL disable math fallback so acceleration failures are explicit during debugging.

#### Scenario: Fused SDPA is required
- **WHEN** SDPA kernel policy is configured at runtime
- **THEN** fused kernel execution SHALL be enabled and math fallback SHALL be disabled

#### Scenario: Unsupported fused path surfaces explicit failure
- **WHEN** runtime conditions cannot satisfy fused SDPA execution
- **THEN** the system SHALL surface a clear startup/runtime error indicating fused-kernel acceleration is unavailable

### Requirement: Compile only UNet for inference optimization
The inference system SHALL apply `torch.compile` only to the UNet module and SHALL keep other pipeline modules uncompiled.

#### Scenario: UNet is compiled
- **WHEN** pipeline components are prepared for inference
- **THEN** `torch.compile` SHALL be applied to UNet before request execution

#### Scenario: Non-UNet components remain eager
- **WHEN** compilation configuration is applied
- **THEN** text encoder, VAE, scheduler, and other non-UNet modules SHALL remain in eager mode

### Requirement: Keep inference fully GPU-resident without CPU offload
The inference system SHALL run without CPU offload for this acceleration path.

#### Scenario: CPU offload is disabled
- **WHEN** inference runtime options are finalized
- **THEN** model execution SHALL remain GPU-resident and SHALL NOT enable CPU offload features
