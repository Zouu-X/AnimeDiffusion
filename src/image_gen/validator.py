"""Validation stage: five completion gates for dataset integrity."""

from __future__ import annotations

import io
import json
import logging
import tarfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from .config import RunConfig

logger = logging.getLogger(__name__)

REQUIRED_METADATA_FIELDS = {
    "prompt", "negative_prompt", "seed", "model_id", "resolution",
    "num_inference_steps", "guidance_scale", "scheduler",
}


@dataclass
class GateResult:
    """Result of a single validation gate."""

    name: str
    passed: bool
    details: str = ""
    failures: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Overall validation report across all gates."""

    passed: bool
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate(config: RunConfig) -> ValidationReport:
    """Run all validation gates and return a report."""
    # Extract all samples from shards
    samples = _extract_all_samples(config)

    gates = [
        _gate_count(samples, config),
        _gate_image_integrity(samples, config),
        _gate_metadata(samples),
        _gate_coverage(samples, config),
    ]

    overall = all(g.passed for g in gates)
    if not overall:
        gates.append(GateResult(
            name="completion",
            passed=False,
            details="One or more validation gates failed",
        ))
    else:
        gates.append(GateResult(
            name="completion",
            passed=True,
            details="All validation gates passed",
        ))

    report = ValidationReport(passed=overall, gates=gates)
    _write_report(report, config)
    return report


def _extract_all_samples(config: RunConfig) -> dict[str, dict]:
    """Extract sample data from all tar shards.

    Returns {sample_id: {"png_data": bytes, "metadata": dict}}.
    """
    samples: dict[str, dict] = {}
    shard_files = sorted(config.shards_dir.glob("anime-face-*.tar"))

    for shard_path in shard_files:
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getmembers()
            png_members = {m.name: m for m in members if m.name.endswith(".png")}
            json_members = {m.name: m for m in members if m.name.endswith(".json")}

            for png_name, png_member in png_members.items():
                sample_id = png_name.removesuffix(".png")
                json_name = f"{sample_id}.json"

                entry: dict = {"png_data": None, "metadata": None}

                f = tar.extractfile(png_member)
                if f:
                    entry["png_data"] = f.read()

                if json_name in json_members:
                    f = tar.extractfile(json_members[json_name])
                    if f:
                        entry["metadata"] = json.loads(f.read())

                samples[sample_id] = entry

    logger.info("Extracted %d samples from %d shards", len(samples), len(shard_files))
    return samples


def _gate_count(samples: dict, config: RunConfig) -> GateResult:
    """Gate 1: Verify exactly target_count unique samples."""
    count = len(samples)
    passed = count == config.target_count
    return GateResult(
        name="count",
        passed=passed,
        details=f"Found {count} samples, expected {config.target_count}",
        failures=[] if passed else [f"Count mismatch: {count} != {config.target_count}"],
    )


def _gate_image_integrity(samples: dict, config: RunConfig) -> GateResult:
    """Gate 2: Verify all PNGs decode and are the expected resolution."""
    failures: list[str] = []
    for sample_id, entry in samples.items():
        png_data = entry.get("png_data")
        if not png_data:
            failures.append(f"{sample_id}: missing PNG data")
            continue
        try:
            img = Image.open(io.BytesIO(png_data))
            img.verify()
            # Re-open after verify (verify can close the file)
            img = Image.open(io.BytesIO(png_data))
            w, h = img.size
            if w != config.resolution or h != config.resolution:
                failures.append(f"{sample_id}: size {w}x{h}, expected {config.resolution}x{config.resolution}")
        except Exception as e:
            failures.append(f"{sample_id}: decode error: {e}")

    passed = len(failures) == 0
    return GateResult(
        name="image_integrity",
        passed=passed,
        details=f"Checked {len(samples)} images, {len(failures)} failures",
        failures=failures[:100],  # Cap failure list
    )


def _gate_metadata(samples: dict) -> GateResult:
    """Gate 3: Verify all samples have required metadata fields."""
    failures: list[str] = []
    for sample_id, entry in samples.items():
        meta = entry.get("metadata")
        if meta is None:
            failures.append(f"{sample_id}: missing metadata JSON")
            continue
        missing = REQUIRED_METADATA_FIELDS - meta.keys()
        if missing:
            failures.append(f"{sample_id}: missing fields {missing}")

    passed = len(failures) == 0
    return GateResult(
        name="metadata",
        passed=passed,
        details=f"Checked {len(samples)} metadata files, {len(failures)} failures",
        failures=failures[:100],
    )


def _gate_coverage(samples: dict, config: RunConfig) -> GateResult:
    """Gate 4: Verify no gaps or duplicates in sample ID range."""
    num_shards = config.target_count // config.shard_size
    remainder = config.target_count % config.shard_size
    total_shards = num_shards + (1 if remainder else 0)

    expected_ids: set[str] = set()
    for shard_id in range(total_shards):
        shard_count = config.shard_size if shard_id < num_shards else remainder
        for r in range(shard_count):
            expected_ids.add(f"s{shard_id:04d}-r{r:04d}")

    actual_ids = set(samples.keys())
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids

    failures: list[str] = []
    if missing:
        failures.append(f"Missing {len(missing)} samples: {sorted(missing)[:20]}...")
    if extra:
        failures.append(f"Extra {len(extra)} samples: {sorted(extra)[:20]}...")

    passed = len(failures) == 0
    return GateResult(
        name="coverage",
        passed=passed,
        details=f"Expected {len(expected_ids)} IDs, found {len(actual_ids)}",
        failures=failures,
    )


def _write_report(report: ValidationReport, config: RunConfig) -> None:
    """Write validation_report.json."""
    out_path = config.report_dir / "validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Validation report written to %s (passed=%s)", out_path, report.passed)
