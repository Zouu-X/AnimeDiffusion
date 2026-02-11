"""Test: run generator twice with same seed -> identical JSONL output."""

import tempfile
from pathlib import Path

from src.random_prompt.generator import run


def test_deterministic_output():
    """Two runs with the same seed must produce byte-identical JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1"
        out2 = Path(tmpdir) / "run2"

        run(seed=42, count=100, shards=2, output_dir=out1)
        run(seed=42, count=100, shards=2, output_dir=out2)

        jsonl1 = (out1 / "prompts.jsonl").read_text()
        jsonl2 = (out2 / "prompts.jsonl").read_text()

        assert jsonl1 == jsonl2, "Outputs differ between runs with same seed"
        assert len(jsonl1.strip().splitlines()) == 100


def test_different_seeds_differ():
    """Different seeds must produce different output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1"
        out2 = Path(tmpdir) / "run2"

        run(seed=42, count=200, shards=1, output_dir=out1)
        run(seed=99, count=200, shards=1, output_dir=out2)

        jsonl1 = (out1 / "prompts.jsonl").read_text()
        jsonl2 = (out2 / "prompts.jsonl").read_text()

        assert jsonl1 != jsonl2, "Different seeds produced identical output"
