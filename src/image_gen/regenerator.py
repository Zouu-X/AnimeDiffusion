"""Regeneration stage: re-generate non-frontal images identified by face_wash."""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import tempfile
from pathlib import Path

import torch

from .config import RunConfig
from .inference import (
    _is_oom_error,
    _resolve_runtime_batch_size,
    _trim_prompt_to_limit,
)

logger = logging.getLogger(__name__)


def load_regen_manifest(manifest_path: Path) -> list[dict]:
    """Read JSONL manifest from face_wash, return list of entry dicts."""
    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "sample_id" not in obj:
                logger.warning("Line %d missing 'sample_id', skipping: %s", lineno, line)
                continue
            entries.append(obj)
    return entries


def _load_regen_progress(path: Path) -> set[str]:
    """Load set of already-regenerated sample IDs from regen_progress.json."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("regenerated_samples", []))
    return set()


def _save_regen_progress(path: Path, done_ids: set[str]) -> None:
    """Atomically write regen_progress.json via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"regenerated_samples": sorted(done_ids)}
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _new_seed(current_seed: int | None) -> int:
    """Generate a new random seed guaranteed to differ from current_seed."""
    seed = random.randint(0, 2**32 - 1)
    if seed == current_seed:
        seed = random.randint(0, 2**32 - 1)
    return seed


def regenerate(config: RunConfig, pipeline, manifest_path: Path) -> None:
    """Re-generate images listed in manifest with new random seeds."""
    config.workspace_dir.mkdir(parents=True, exist_ok=True)

    entries = load_regen_manifest(manifest_path)
    progress_path = config.report_dir / "regen_progress.json"
    done_ids = _load_regen_progress(progress_path)

    pending = [e for e in entries if e["sample_id"] not in done_ids]
    logger.info(
        "Regenerating %d samples (%d already done)",
        len(pending),
        len(done_ids),
    )

    if not pending:
        logger.info("Regeneration complete: 0 samples (all already done)")
        return

    runtime_batch_size = _resolve_runtime_batch_size(config, pipeline)

    text_tokenizers = [
        tok
        for tok in [
            getattr(pipeline, "tokenizer", None),
            getattr(pipeline, "tokenizer_2", None),
        ]
        if tok is not None
    ]

    batch_num = 0
    i = 0
    total = len(pending)
    regenerated = 0

    while i < len(pending):
        batch_entries = pending[i : i + runtime_batch_size]
        batch_num += 1

        # Load workspace JSON for each entry in the batch
        prepared_batch = []
        for entry in batch_entries:
            sample_id = entry["sample_id"]
            json_path = config.workspace_dir / f"{sample_id}.json"
            if not json_path.exists():
                logger.warning("Workspace JSON missing for %s, skipping", sample_id)
                continue
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
            current_seed = meta.get("seed")
            new_seed = _new_seed(current_seed)
            pos_prompt, _ = _trim_prompt_to_limit(meta.get("prompt", ""), text_tokenizers)
            neg_prompt, _ = _trim_prompt_to_limit(meta.get("negative_prompt", ""), text_tokenizers)
            prepared_batch.append((sample_id, pos_prompt, neg_prompt, new_seed, meta))

        if not prepared_batch:
            i += len(batch_entries)
            continue

        micro_batch_size = len(prepared_batch)
        j = 0
        while j < len(prepared_batch):
            micro_batch = prepared_batch[j : j + micro_batch_size]
            generators = [
                torch.Generator(device="cpu").manual_seed(new_seed)
                for _, _, _, new_seed, _ in micro_batch
            ]

            try:
                results = pipeline(
                    prompt=[pos_prompt for _, pos_prompt, _, _, _ in micro_batch],
                    negative_prompt=[neg_prompt for _, _, neg_prompt, _, _ in micro_batch],
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    height=config.resolution,
                    width=config.resolution,
                    generator=generators,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                if micro_batch_size == 1:
                    raise
                reduced = max(1, micro_batch_size // 2)
                logger.warning(
                    "CUDA OOM at micro-batch %d; retrying with %d",
                    micro_batch_size,
                    reduced,
                )
                micro_batch_size = reduced
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for (sample_id, pos_prompt, neg_prompt, new_seed, orig_meta), image in zip(
                micro_batch, results.images
            ):
                png_path = config.workspace_dir / f"{sample_id}.png"
                json_path = config.workspace_dir / f"{sample_id}.json"

                image.save(png_path)

                metadata = {
                    "prompt": pos_prompt,
                    "negative_prompt": neg_prompt,
                    "seed": new_seed,
                    "model_id": orig_meta.get("model_id", config.model_id),
                    "num_inference_steps": config.num_inference_steps,
                    "guidance_scale": config.guidance_scale,
                    "scheduler": config.scheduler,
                    "resolution": config.resolution,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f)

                done_ids.add(sample_id)

            regenerated += len(micro_batch)
            _save_regen_progress(progress_path, done_ids)
            j += len(micro_batch)
            del results

        i += len(batch_entries)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Batch %d: regenerated %d/%d samples",
            batch_num,
            regenerated,
            total,
        )

    logger.info("Regeneration complete: %d samples", regenerated)
