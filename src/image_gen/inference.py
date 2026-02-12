"""Inference stage: load model, batched image generation with checkpointing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionPipeline
from PIL import Image

from .config import RunConfig
from .progress import ProgressState

logger = logging.getLogger(__name__)

# Checkpoint filename
_CKPT_FILENAME = "wd-1-4-anime_e1.ckpt"
_CKPT_URL = f"https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/{_CKPT_FILENAME}"


def _select_device() -> tuple[torch.device, torch.dtype]:
    """Pick the best available device and matching dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def load_pipeline(config: RunConfig) -> StableDiffusionPipeline:
    """Load the Waifu Diffusion v1.4 checkpoint."""
    device, dtype = _select_device()
    logger.info("Loading model %s on %s (dtype=%s)", config.model_id, device, dtype)

    pipe = StableDiffusionPipeline.from_single_file(
        _CKPT_URL,
        torch_dtype=dtype,
    )

    # Apply scheduler
    if config.scheduler == "EulerAncestralDiscreteScheduler":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info("Model loaded successfully")
    return pipe


def _load_prompts(config: RunConfig) -> list[dict]:
    """Load prompt records from JSONL up to target_count."""
    records: list[dict] = []
    with open(config.prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= config.target_count:
                break
    return records


def generate(config: RunConfig, pipeline: StableDiffusionPipeline, progress: ProgressState) -> None:
    """Generate images in batches, saving PNGs + JSON sidecars to workspace."""
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    records = _load_prompts(config)
    device = pipeline.device

    # Filter out already-finalized samples
    pending = [r for r in records if not progress.is_sample_done(r["id"])]
    total = len(records)
    done = total - len(pending)
    logger.info("Resuming: %d/%d already done, %d remaining", done, total, len(pending))

    batch_num = 0
    for i in range(0, len(pending), config.batch_size):
        batch = pending[i : i + config.batch_size]
        batch_num += 1

        # Build generators with per-sample seeds
        generators = [
            torch.Generator(device="cpu").manual_seed(rec["seed"])
            for rec in batch
        ]

        # Run inference
        results = pipeline(
            prompt=[rec["positive_prompt"] for rec in batch],
            negative_prompt=[rec["negative_prompt"] for rec in batch],
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.resolution,
            width=config.resolution,
            generator=generators,
        )

        # Save outputs
        for rec, image in zip(batch, results.images):
            sample_id = rec["id"]
            png_path = config.workspace_dir / f"{sample_id}.png"
            json_path = config.workspace_dir / f"{sample_id}.json"

            image.save(png_path)

            metadata = {
                "prompt": rec["positive_prompt"],
                "negative_prompt": rec["negative_prompt"],
                "seed": rec["seed"],
                "model_id": config.model_id,
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
                "scheduler": config.scheduler,
                "resolution": config.resolution,
            }
            with open(json_path, "w") as f:
                json.dump(metadata, f)

            progress.mark_sample_done(sample_id)

        # Checkpoint after each batch
        progress.save()
        done += len(batch)
        logger.info("Batch %d: generated %d/%d samples", batch_num, done, total)

    logger.info("Generation complete: %d samples", total)
