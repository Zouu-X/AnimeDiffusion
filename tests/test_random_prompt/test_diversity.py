"""Test: diversity thresholds and finalization guard."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.random_prompt.diversity import DedupTracker, DiversityTracker
from src.random_prompt.exporter import Exporter, FinalizationError
from src.random_prompt.schema import PromptComponents, PromptRecord


def _make_components(**overrides) -> PromptComponents:
    defaults = dict(
        subject="1girl, teenage girl",
        face_shape="round face",
        eyes="blue eyes",
        nose="small nose",
        mouth="small mouth",
        hair_color="blonde hair",
        hair_style="long hair",
        expression="smile",
        accessories="",
        style_modifiers=["masterpiece"],
        lighting="natural lighting",
        background="simple background",
    )
    defaults.update(overrides)
    return PromptComponents(**defaults)


def test_diversity_tracker_counts():
    """DiversityTracker should count unique values."""
    tracker = DiversityTracker()
    tracker.record(_make_components(eyes="blue eyes"))
    tracker.record(_make_components(eyes="red eyes"))
    tracker.record(_make_components(eyes="blue eyes"))  # duplicate

    results = tracker.check_thresholds()
    # eyes should have 2 unique values
    assert results["eyes"][1] == 2


def test_below_threshold_blocks_finalization():
    """When diversity thresholds are not met, finalization must raise."""
    tracker = DiversityTracker(
        thresholds={"eyes": 5, "hair_color": 5, "hair_style": 5, "expression": 3, "accessories": 3}
    )
    # Only record 2 unique eyes — below threshold of 5
    tracker.record(_make_components(eyes="blue eyes", hair_color="blonde hair", hair_style="long hair", expression="smile"))
    tracker.record(_make_components(eyes="red eyes", hair_color="black hair", hair_style="short hair", expression="serious"))

    assert not tracker.all_thresholds_met()

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = Exporter(Path(tmpdir))
        exporter.begin()
        # Write a dummy record
        exporter.write_record(PromptRecord(
            id="s0000-r0000",
            positive_prompt="test prompt",
            negative_prompt="test negative",
            shard_id=0,
            seed=42,
        ))

        with pytest.raises(FinalizationError):
            exporter.finalize(tracker, seed=42, config_hash="abc", total_requested=1)


def test_above_threshold_allows_finalization():
    """When diversity thresholds are met, finalization should succeed."""
    tracker = DiversityTracker(
        thresholds={"eyes": 2, "hair_color": 2, "hair_style": 2, "expression": 2, "accessories": 2}
    )
    # Record enough unique values
    tracker.record(_make_components(eyes="blue eyes", hair_color="blonde hair", hair_style="long hair",
                                    expression="smile", accessories="earrings"))
    tracker.record(_make_components(eyes="red eyes", hair_color="black hair", hair_style="short hair",
                                    expression="serious", accessories="headband"))

    assert tracker.all_thresholds_met()

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = Exporter(Path(tmpdir))
        exporter.begin()
        exporter.write_record(PromptRecord(
            id="s0000-r0000",
            positive_prompt="test prompt",
            negative_prompt="test negative",
            shard_id=0,
            seed=42,
        ))

        manifest_path = exporter.finalize(tracker, seed=42, config_hash="abc", total_requested=1)
        assert manifest_path.exists()


def test_dedup_tracker():
    """DedupTracker should detect near-duplicate prompts."""
    dedup = DedupTracker()
    assert not dedup.is_duplicate("masterpiece, blue eyes, smile")
    assert dedup.is_duplicate("masterpiece, blue eyes, smile")  # exact dup
    assert dedup.is_duplicate("smile, blue eyes, masterpiece")  # reordered
    assert not dedup.is_duplicate("masterpiece, red eyes, smile")  # different


def test_dedup_case_insensitive():
    """DedupTracker canonicalization is case-insensitive."""
    dedup = DedupTracker()
    assert not dedup.is_duplicate("Masterpiece, Blue Eyes")
    assert dedup.is_duplicate("masterpiece, blue eyes")
