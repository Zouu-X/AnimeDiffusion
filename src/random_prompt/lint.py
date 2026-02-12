"""Lint rules, validation, and rejection logic."""

from .schema import PromptComponents

# Tokens that should NEVER appear in positive prompts (quality degraders).
BANNED_POSITIVE_TOKENS = {
    "lowres", "bad anatomy", "bad hands", "text", "error",
    "worst quality", "low quality", "normal quality", "jpeg artifacts",
    "signature", "watermark", "username", "blurry",
    "ugly", "duplicate", "mutation", "deformed", "disfigured",
}

# Minimum number of comma-separated tokens in a positive prompt.
MIN_TOKEN_COUNT = 8

# Pose bounds.
YAW_RANGE = (-15.0, 15.0)
PITCH_RANGE = (-15.0, 15.0)


def lint_components(components: PromptComponents) -> list[str]:
    """Validate components and return a list of violation descriptions.
    Empty list means valid."""
    violations = []

    if not components.subject:
        violations.append("missing subject")

    if not (YAW_RANGE[0] <= components.yaw <= YAW_RANGE[1]):
        violations.append(f"yaw {components.yaw} out of range {YAW_RANGE}")

    if not (PITCH_RANGE[0] <= components.pitch <= PITCH_RANGE[1]):
        violations.append(f"pitch {components.pitch} out of range {PITCH_RANGE}")

    if not components.eyes:
        violations.append("missing eyes")
    if not components.hair_color:
        violations.append("missing hair_color")
    if not components.hair_style:
        violations.append("missing hair_style")
    if not components.expression:
        violations.append("missing expression")

    return violations


def lint_positive_prompt(positive_prompt: str) -> list[str]:
    """Validate assembled positive prompt. Returns violation list."""
    violations = []
    tokens = [t.strip().lower() for t in positive_prompt.split(",")]

    for banned in BANNED_POSITIVE_TOKENS:
        if banned in tokens:
            violations.append(f"banned token in positive prompt: '{banned}'")

    if len(tokens) < MIN_TOKEN_COUNT:
        violations.append(
            f"too few tokens: {len(tokens)} < {MIN_TOKEN_COUNT}"
        )

    return violations


def lint_semantic(positive_prompt: str) -> list[str]:
    """Check assembled prompt for semantic contradictions."""
    violations = []
    tokens = [t.strip().lower() for t in positive_prompt.split(",")]

    has_closed_eyes = "closed eyes" in tokens
    looking_tokens = {"looking to the side", "looking away", "looking up", "looking down"}
    if has_closed_eyes and looking_tokens & set(tokens):
        violations.append("contradiction: closed eyes with looking direction")

    has_closed_mouth = "closed mouth" in tokens
    if has_closed_mouth and "laughing" in tokens:
        violations.append("contradiction: closed mouth with laughing")

    return violations


def lint_record_fields(record_dict: dict) -> list[str]:
    """Validate that all required fields are present and non-empty."""
    violations = []
    required = ["id", "positive_prompt", "negative_prompt", "shard_id", "seed"]
    for field in required:
        if field not in record_dict or record_dict[field] is None:
            violations.append(f"missing field: {field}")
        elif isinstance(record_dict[field], str) and not record_dict[field].strip():
            violations.append(f"empty field: {field}")
    return violations
