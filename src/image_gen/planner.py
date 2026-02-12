"""Plan stage: load prompts, validate, and build sample manifest."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from .config import RunConfig

logger = logging.getLogger(__name__)


def plan(config: RunConfig) -> dict:
    """Read prompts.jsonl, validate record count and structure, write sample manifest.

    Returns the manifest dict.
    """
    records = _load_prompts(config.prompts_path, config.target_count)
    _validate_records(records, config)
    manifest = _build_manifest(records, config)
    _write_manifest(manifest, config)
    return manifest


def _load_prompts(path: Path, target_count: int) -> list[dict]:
    """Load all prompt records from JSONL."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= target_count:
                break
    logger.info("Loaded %d prompt records from %s", len(records), path)
    return records


def _validate_records(records: list[dict], config: RunConfig) -> None:
    """Validate record count, required fields, and seed uniqueness."""
    if len(records) != config.target_count:
        raise ValueError(
            f"Expected {config.target_count} records, got {len(records)}"
        )

    required_fields = {"id", "positive_prompt", "negative_prompt", "shard_id", "seed"}
    seeds: set[int] = set()
    ids: set[str] = set()
    shard_counts: dict[int, int] = defaultdict(int)

    for i, rec in enumerate(records):
        missing = required_fields - rec.keys()
        if missing:
            raise ValueError(f"Record {i} missing fields: {missing}")
        if rec["id"] in ids:
            raise ValueError(f"Duplicate sample ID: {rec['id']}")
        if rec["seed"] in seeds:
            raise ValueError(f"Duplicate seed at record {i}: {rec['seed']}")
        ids.add(rec["id"])
        seeds.add(rec["seed"])
        shard_counts[rec["shard_id"]] += 1

    # Verify shard alignment
    num_shards = config.target_count // config.shard_size
    remainder = config.target_count % config.shard_size
    expected_shards = num_shards + (1 if remainder else 0)
    if len(shard_counts) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} shards, found {len(shard_counts)}"
        )
    for shard_id in range(expected_shards):
        expected = config.shard_size if shard_id < num_shards else remainder
        actual = shard_counts.get(shard_id, 0)
        if actual != expected:
            raise ValueError(
                f"Shard {shard_id}: expected {expected} records, got {actual}"
            )

    logger.info(
        "Validation passed: %d records, %d shards, %d unique seeds",
        len(records), len(shard_counts), len(seeds),
    )


def _build_manifest(records: list[dict], config: RunConfig) -> dict:
    """Build sample manifest summary."""
    shard_ids = sorted({r["shard_id"] for r in records})
    return {
        "total_samples": len(records),
        "shard_count": len(shard_ids),
        "shard_size": config.shard_size,
        "shard_ids": shard_ids,
        "seed_range": [min(r["seed"] for r in records), max(r["seed"] for r in records)],
        "sample_id_range": [records[0]["id"], records[-1]["id"]],
    }


def _write_manifest(manifest: dict, config: RunConfig) -> None:
    """Write sample_manifest.json."""
    out_path = config.report_dir / "sample_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Sample manifest written to %s", out_path)
