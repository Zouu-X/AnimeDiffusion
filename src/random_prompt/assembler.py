"""Prompt assembly: PromptComponents -> (positive_prompt, negative_prompt)."""

from .schema import PromptComponents


# Waifu Diffusion v1.4 tag ordering:
# quality tags -> subject -> appearance -> expression -> accessories -> pose -> lighting -> background
_QUALITY_TAGS_IN_POSITIVE = {"masterpiece", "best quality", "highres", "absurdres", "ultra-detailed"}
_FRONTAL_POSE_TAGS = ("frontal face", "looking at viewer")


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

    # 5) Optional accessories
    if components.accessories:
        tokens.append(components.accessories)

    # 6) Constant frontal pose tags
    tokens.extend(_FRONTAL_POSE_TAGS)

    # 7) Lighting
    if components.lighting:
        tokens.append(components.lighting)

    # 8) Background
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

    style_tags = {t.strip().lower() for t in components.style_modifiers}

    # Add face-focus suppressions only for tight portrait framing.
    if "face_focus" in conditional and {"close-up", "portrait"} & style_tags:
        parts.extend(conditional["face_focus"])

    # Add eye artifact suppression only when eyes are explicitly emphasized.
    detailed_eye_traits = {"heterochromia", "glowing eyes", "sparkling eyes", "empty eyes"}
    if "detailed_eyes" in conditional and (
        "detailed eyes" in style_tags or components.eyes.strip().lower() in detailed_eye_traits
    ):
        parts.extend(conditional["detailed_eyes"])

    # Add general quality suppression
    if "quality" in conditional:
        parts.extend(conditional["quality"])

    # Always reinforce frontal, non-profile composition.
    if "anti_side_view" in conditional:
        parts.extend(conditional["anti_side_view"])

    # Always suppress extreme close framing.
    if "extreme_closeup" in conditional:
        parts.extend(conditional["extreme_closeup"])

    return ", ".join(parts)
