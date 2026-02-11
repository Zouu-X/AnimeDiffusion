"""Config loading and defaults."""

from pathlib import Path

import yaml

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "random_prompt"

DEFAULT_SEED = 42
DEFAULT_COUNT = 100_000
DEFAULT_SHARDS = 100
PILOT_COUNT = 1_000
PILOT_SHARDS = 1


def load_vocab(config_dir: Path | None = None) -> dict:
    """Load and validate the vocabulary config."""
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    vocab_path = config_dir / "vocab.yaml"
    with open(vocab_path) as f:
        vocab = yaml.safe_load(f)

    required_components = [
        "age_style", "face_shape", "eyes", "nose", "mouth",
        "hair", "expression", "style_modifiers", "lighting", "background",
    ]
    for comp in required_components:
        if comp not in vocab:
            raise ValueError(f"Missing required component '{comp}' in vocab.yaml")
        if "options" not in vocab[comp]:
            raise ValueError(f"Component '{comp}' missing 'options' list")
        if not vocab[comp]["options"]:
            raise ValueError(f"Component '{comp}' has empty options list")

    # Validate age_style is restricted to teen/adult
    age_values = {opt["value"] for opt in vocab["age_style"]["options"]}
    if not age_values <= {"teen", "adult"}:
        raise ValueError(f"age_style must only contain 'teen'/'adult', got {age_values}")

    return vocab


def load_negative_prompts(config_dir: Path | None = None) -> dict:
    """Load negative prompt templates."""
    config_dir = config_dir or _DEFAULT_CONFIG_DIR
    neg_path = config_dir / "negative_prompts.yaml"
    with open(neg_path) as f:
        return yaml.safe_load(f)
