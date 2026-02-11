"""JSONL export, manifest, and finalization guard."""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .diversity import DiversityTracker
from .schema import PromptRecord


class FinalizationError(Exception):
    """Raised when export is blocked due to unmet diversity thresholds."""


class Exporter:
    """Handles JSONL output and manifest generation."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._output_dir / "prompts.jsonl"
        self._manifest_path = self._output_dir / "manifest.json"
        self._records_written = 0

    @property
    def jsonl_path(self) -> Path:
        return self._jsonl_path

    def begin(self) -> None:
        """Open/truncate output file for writing."""
        # Truncate to start fresh
        self._jsonl_path.write_text("")
        self._records_written = 0

    def write_record(self, record: PromptRecord) -> None:
        """Append a single record as a JSONL line."""
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        self._records_written += 1

    def write_records(self, records: list[PromptRecord]) -> None:
        """Append a batch of records."""
        with open(self._jsonl_path, "a") as f:
            for record in records:
                f.write(json.dumps(asdict(record)) + "\n")
                self._records_written += 1

    def finalize(
        self,
        diversity_tracker: DiversityTracker,
        seed: int,
        config_hash: str,
        total_requested: int,
    ) -> Path:
        """Write manifest and finalize. Raises FinalizationError if
        diversity thresholds are not met."""
        if not diversity_tracker.all_thresholds_met():
            results = diversity_tracker.check_thresholds()
            failures = {
                k: {"unique": v[1], "required": v[2]}
                for k, v in results.items() if not v[0]
            }
            raise FinalizationError(
                f"Diversity thresholds not met: {failures}"
            )

        # Compute file hash
        content = self._jsonl_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        manifest = {
            "version": "0.1.0",
            "seed": seed,
            "config_hash": config_hash,
            "total_records": self._records_written,
            "total_requested": total_requested,
            "output_file": self._jsonl_path.name,
            "output_hash": file_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return self._manifest_path
