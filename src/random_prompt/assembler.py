"""Prompt assembly: PromptComponents -> (positive_prompt, negative_prompt)."""

from .schema import PromptComponents


# Waifu Diffusion v1.4 tag ordering:
# quality tags -> subject -> appearance -> expression -> pose -> lighting -> background
_QUALITY_TAGS_IN_POSITIVE = {"masterpiece", "best quality", "highres", "absurdres", "ultra-detailed"}


def _pose_descriptor(yaw: float, pitch: float) -> str:
    """Convert yaw/pitch to a single-direction textual pose descriptor.

    Picks the dominant axis (larger absolute value) and emits one tag.
    If both |yaw| and |pitch| <= 3, emits "facing viewer".
    """
    if abs(yaw) <= 3 and abs(pitch) <= 3:
        return "facing viewer"
    # Choose the dominant axis
    if abs(yaw) >= abs(pitch):
        if yaw > 0:
            return "looking to the side"
        else:
            return "looking away"
    else:
        if pitch > 0:
            return "looking up"
        else:
            return "looking down"


def assemble_positive(components: PromptComponents) -> str:
    """Build a positive prompt in Waifu Diffusion v1.4 tag style."""
    tokens: list[str] = []

    # 1) Quality/style modifiers first (sorted for consistency)
    quality_first = []
    other_style = []
    for mod in components.style_modifiers:
        if mod in _QUALITY_TAGS_IN_POSITIVE:
            quality_first.append(mod)
        else:
            other_style.append(mod)
    tokens.extend(quality_first)
    tokens.extend(other_style)

    # 2) Subject (e.g. "1girl, teenage girl" or "1boy, young man")
    for tag in components.subject.split(", "):
        tokens.append(tag)

    # 3) Appearance: hair color, hair style, eyes, face, nose, mouth
    tokens.append(components.hair_color)
    tokens.append(components.hair_style)
    tokens.append(components.eyes)
    tokens.append(components.face_shape)
    tokens.append(components.nose)
    tokens.append(components.mouth)

    # 4) Expression
    tokens.append(components.expression)

    # 5) Pose
    pose = _pose_descriptor(components.yaw, components.pitch)
    tokens.append(pose)

    # 6) Lighting
    if components.lighting:
        tokens.append(components.lighting)

    # 7) Background
    if components.background:
        tokens.append(components.background)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return ", ".join(deduped)


def assemble_negative(components: PromptComponents, negative_config: dict) -> str:
    """Build a negative prompt from base template + conditional additions."""
    parts = [negative_config["base_negative"]]

    conditional = negative_config.get("conditional_negatives", {})

    # Always add face focus negatives for face generation
    if "face_focus" in conditional:
        parts.extend(conditional["face_focus"])

    # Add eye artifact suppression if detailed eyes are involved
    if "detailed_eyes" in conditional:
        parts.extend(conditional["detailed_eyes"])

    # Add general quality suppression
    if "quality" in conditional:
        parts.extend(conditional["quality"])

    return ", ".join(parts)
