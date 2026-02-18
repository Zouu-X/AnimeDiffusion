"""Typed prompt schema components."""

from dataclasses import dataclass, field


@dataclass
class PromptComponents:
    """Individual sampled components that make up a prompt."""

    subject: str  # e.g. "1girl, teenage girl" or "1boy, young man"
    face_shape: str
    eyes: str
    nose: str
    mouth: str
    hair_color: str
    hair_style: str
    expression: str
    accessories: str = ""
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
