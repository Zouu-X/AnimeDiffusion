"""Typed prompt schema components."""

from dataclasses import dataclass, field


@dataclass
class PromptComponents:
    """Individual sampled components that make up a prompt."""

    age_style: str  # "teen" | "adult"
    face_shape: str
    eyes: str
    nose: str
    mouth: str
    hair: str
    expression: str
    yaw: float  # [-15, 15]
    pitch: float  # [-15, 15]
    style_modifiers: list[str] = field(default_factory=list)
    lighting: str = ""
    background: str = ""


@dataclass
class PromptRecord:
    """A single generated prompt record with positive and negative prompts."""

    id: str  # e.g. "s0042-r0817"
    positive_prompt: str
    negative_prompt: str
    shard_id: int
    seed: int
