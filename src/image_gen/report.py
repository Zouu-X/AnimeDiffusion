"""Run summary report generation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig
from .validator import ValidationReport

logger = logging.getLogger(__name__)


def write_run_report(
    config: RunConfig,
    validation: ValidationReport,
    elapsed_seconds: float | None = None,
) -> Path:
    """Write run_report.json with summary of the run."""
    num_shards = config.target_count // config.shard_size
    remainder = config.target_count % config.shard_size
    total_shards = num_shards + (1 if remainder else 0)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_count": config.target_count,
        "shard_count": total_shards,
        "shard_size": config.shard_size,
        "model_id": config.model_id,
        "validation_passed": validation.passed,
        "validation_summary": {
            g.name: {"passed": g.passed, "details": g.details}
            for g in validation.gates
        },
    }

    if elapsed_seconds is not None and elapsed_seconds > 0:
        report["elapsed_seconds"] = round(elapsed_seconds, 2)
        report["throughput_samples_per_sec"] = round(
            config.target_count / elapsed_seconds, 2
        )

    if not validation.passed:
        report["failure_breakdown"] = {
            g.name: g.failures
            for g in validation.gates
            if not g.passed and g.failures
        }

    out_path = config.report_dir / "run_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Run report written to %s", out_path)
    return out_path
