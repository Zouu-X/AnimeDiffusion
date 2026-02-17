"""Test: verify subject diversity, pose direction, hair fields, and output field completeness."""

import json
import tempfile
from pathlib import Path

from src.random_prompt.assembler import assemble_negative, assemble_positive
from src.random_prompt.generator import run
from src.random_prompt.lint import YAW_RANGE, PITCH_RANGE
from src.random_prompt.sampler import ComponentSampler
from src.random_prompt.config import load_negative_prompts, load_vocab
from src.random_prompt.schema import PromptComponents
import random


def test_subject_diversity():
    """Subject sampler must produce both '1girl' and '1boy' subjects."""
    vocab = load_vocab()
    sampler = ComponentSampler(vocab)
    rng = random.Random(123)
    subjects = set()
    for _ in range(500):
        components = sampler.sample(rng)
        subjects.add(components.subject)
    subject_str = " ".join(subjects)
    assert "1girl" in subject_str, "Expected at least one '1girl' subject"
    assert "1boy" in subject_str, "Expected at least one '1boy' subject"


def test_pose_within_bounds():
    """yaw and pitch must be within [-15, 15]."""
    vocab = load_vocab()
    sampler = ComponentSampler(vocab)
    rng = random.Random(456)
    for _ in range(500):
        components = sampler.sample(rng)
        assert YAW_RANGE[0] <= components.yaw <= YAW_RANGE[1], (
            f"yaw out of range: {components.yaw}"
        )
        assert PITCH_RANGE[0] <= components.pitch <= PITCH_RANGE[1], (
            f"pitch out of range: {components.pitch}"
        )


def test_single_pose_direction():
    """Assembled prompts must contain at most one 'looking' direction token."""
    vocab = load_vocab()
    sampler = ComponentSampler(vocab)
    rng = random.Random(789)
    looking_tokens = {"looking to the side", "looking away", "looking up", "looking down"}
    for _ in range(500):
        components = sampler.sample(rng)
        prompt = assemble_positive(components)
        tokens = [t.strip().lower() for t in prompt.split(",")]
        found = [t for t in tokens if t in looking_tokens]
        assert len(found) <= 1, (
            f"Multiple looking directions in prompt: {found}"
        )


def test_hair_has_color():
    """Sampled components must have a non-empty hair_color containing a color keyword."""
    vocab = load_vocab()
    sampler = ComponentSampler(vocab)
    rng = random.Random(321)
    for _ in range(500):
        components = sampler.sample(rng)
        assert components.hair_color, "hair_color must not be empty"
        assert "hair" in components.hair_color.lower(), (
            f"hair_color should contain 'hair': {components.hair_color}"
        )


def test_all_output_fields_present():
    """Every record in output must have all required fields non-empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_output"
        run(seed=42, count=50, shards=1, output_dir=out)

        with open(out / "prompts.jsonl") as f:
            for line_num, line in enumerate(f, 1):
                record = json.loads(line)
                for field in ["id", "positive_prompt", "negative_prompt", "shard_id", "seed"]:
                    assert field in record, f"Line {line_num}: missing field '{field}'"
                    if isinstance(record[field], str):
                        assert record[field].strip(), (
                            f"Line {line_num}: empty field '{field}'"
                        )


def test_no_banned_tokens_in_positive():
    """Positive prompts must not contain banned quality tokens."""
    from src.random_prompt.lint import BANNED_POSITIVE_TOKENS

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test_output"
        run(seed=42, count=100, shards=1, output_dir=out)

        with open(out / "prompts.jsonl") as f:
            for line in f:
                record = json.loads(line)
                tokens = {t.strip().lower() for t in record["positive_prompt"].split(",")}
                for banned in BANNED_POSITIVE_TOKENS:
                    assert banned not in tokens, (
                        f"Banned token '{banned}' found in: {record['positive_prompt']}"
                    )


def test_negative_prompt_conditionals_are_applied_selectively():
    """Conditional negatives should only be added when their trigger tags are present."""
    negative_config = load_negative_prompts()
    components = PromptComponents(
        subject="1girl",
        face_shape="round face",
        eyes="blue eyes",
        nose="small nose",
        mouth="small mouth",
        hair_color="black hair",
        hair_style="long hair",
        expression="smile",
        yaw=0.0,
        pitch=0.0,
        style_modifiers=["masterpiece"],
        lighting="soft lighting",
        background="simple background",
    )
    negative = assemble_negative(components, negative_config)
    tokens = {t.strip() for t in negative.split(",")}

    assert "extra limbs" not in tokens
    assert "bad eyes" not in tokens
    assert "bad hands" in tokens


def test_negative_prompt_adds_closeup_and_detailed_eye_conditionals():
    """Face-focus and eye-specific negatives should appear when trigger tags are present."""
    negative_config = load_negative_prompts()
    components = PromptComponents(
        subject="1girl",
        face_shape="round face",
        eyes="blue eyes",
        nose="small nose",
        mouth="small mouth",
        hair_color="black hair",
        hair_style="long hair",
        expression="smile",
        yaw=0.0,
        pitch=0.0,
        style_modifiers=["close-up", "detailed eyes", "masterpiece"],
        lighting="soft lighting",
        background="simple background",
    )
    negative = assemble_negative(components, negative_config)
    tokens = {t.strip() for t in negative.split(",")}

    assert "extra limbs" in tokens
    assert "bad eyes" in tokens
