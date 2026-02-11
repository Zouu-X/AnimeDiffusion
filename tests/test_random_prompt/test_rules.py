"""Test: verify age buckets, pose range, and output field completeness."""

import json
import tempfile
from pathlib import Path

from src.random_prompt.generator import run
from src.random_prompt.lint import VALID_AGE_STYLES, YAW_RANGE, PITCH_RANGE
from src.random_prompt.sampler import ComponentSampler
from src.random_prompt.config import load_vocab
import random


def test_age_style_restricted():
    """age_style sampler must only produce 'teen' or 'adult'."""
    vocab = load_vocab()
    sampler = ComponentSampler(vocab)
    rng = random.Random(123)
    for _ in range(500):
        components = sampler.sample(rng)
        assert components.age_style in VALID_AGE_STYLES, (
            f"Got invalid age_style: {components.age_style}"
        )


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
