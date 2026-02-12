"""Package stage: write WebDataset .tar shards from workspace files."""

from __future__ import annotations

import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path

from .config import RunConfig
from .progress import ProgressState

logger = logging.getLogger(__name__)


def package(config: RunConfig, progress: ProgressState) -> None:
    """Package workspace PNGs + JSONs into WebDataset tar shards."""
    config.shards_dir.mkdir(parents=True, exist_ok=True)

    num_shards = config.target_count // config.shard_size
    remainder = config.target_count % config.shard_size
    total_shards = num_shards + (1 if remainder else 0)

    for shard_id in range(total_shards):
        if progress.is_shard_done(shard_id):
            logger.info("Shard %d already complete, skipping", shard_id)
            continue

        shard_count = config.shard_size if shard_id < num_shards else remainder
        sample_ids = [
            f"s{shard_id:04d}-r{r:04d}" for r in range(shard_count)
        ]

        # Verify all samples exist in workspace
        missing = []
        for sid in sample_ids:
            png = config.workspace_dir / f"{sid}.png"
            meta = config.workspace_dir / f"{sid}.json"
            if not png.exists() or not meta.exists():
                missing.append(sid)
        if missing:
            logger.error(
                "Shard %d: %d samples missing from workspace, skipping",
                shard_id, len(missing),
            )
            continue

        shard_name = f"anime-face-{shard_id:06d}.tar"
        final_path = config.shards_dir / shard_name
        tmp_path = final_path.with_suffix(".tar.tmp")

        _write_shard(tmp_path, sample_ids, config.workspace_dir)
        os.rename(tmp_path, final_path)
        progress.mark_shard_done(shard_id)
        progress.save()
        logger.info("Shard %d: packaged %d samples -> %s", shard_id, len(sample_ids), shard_name)

    logger.info("Packaging complete: %d shards", total_shards)


def _write_shard(tar_path: Path, sample_ids: list[str], workspace: Path) -> None:
    """Write a single tar shard containing PNGs and JSON sidecars."""
    with tarfile.open(tar_path, "w") as tar:
        for sid in sample_ids:
            png_path = workspace / f"{sid}.png"
            json_path = workspace / f"{sid}.json"
            tar.add(png_path, arcname=f"{sid}.png")
            tar.add(json_path, arcname=f"{sid}.json")
