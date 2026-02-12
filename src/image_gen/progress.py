"""Progress state persistence for resumable runs."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProgressState:
    """Tracks which samples and shards have been completed."""

    path: Path
    finalized_samples: set[str] = field(default_factory=set)
    completed_shards: set[int] = field(default_factory=set)

    def mark_sample_done(self, sample_id: str) -> None:
        self.finalized_samples.add(sample_id)

    def mark_shard_done(self, shard_id: int) -> None:
        self.completed_shards.add(shard_id)

    def is_sample_done(self, sample_id: str) -> bool:
        return sample_id in self.finalized_samples

    def is_shard_done(self, shard_id: int) -> bool:
        return shard_id in self.completed_shards

    def save(self) -> None:
        """Atomically persist state via temp file + rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "finalized_samples": sorted(self.finalized_samples),
            "completed_shards": sorted(self.completed_shards),
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.rename(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: Path) -> ProgressState:
        """Load existing progress or return empty state."""
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(
                path=path,
                finalized_samples=set(data.get("finalized_samples", [])),
                completed_shards=set(data.get("completed_shards", [])),
            )
        return cls(path=path)
