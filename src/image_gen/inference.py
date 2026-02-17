"""Inference stage: load model, batched image generation with checkpointing."""

from __future__ import annotations

import json
import logging

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline

from .config import RunConfig
from .progress import ProgressState

logger = logging.getLogger(__name__)


def _select_device() -> tuple[torch.device, torch.dtype]:
    """Pick the best available device and matching dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_pipeline(config: RunConfig) -> StableDiffusionXLPipeline:
    """Load the Illustrious XL checkpoint."""
    device, dtype = _select_device()
    logger.info("Loading checkpoint %s on %s (dtype=%s)", config.checkpoint_path, device, dtype)

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(config.checkpoint_path),
        torch_dtype=dtype,
    )

    # Apply scheduler
    if config.scheduler == "EulerAncestralDiscreteScheduler":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

    pipe = pipe.to(device)
    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as exc:
            logger.warning("xformers not available, continuing without it: %s", exc)

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


def _count_tokens(tokenizer: object, text: str) -> int:
    """Return token count for a text string using a tokenizer-compatible API."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)  # type: ignore[misc]
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def _effective_max_length(tokenizer: object) -> int:
    """Resolve a practical max token length for CLIP-style tokenizers."""
    max_len = int(getattr(tokenizer, "model_max_length", 77))
    # Some tokenizers expose very large sentinel values; SD/SDXL CLIP uses 77.
    if max_len > 4096:
        return 77
    return max_len


def _trim_prompt_to_limit(text: str, tokenizers: list[object]) -> tuple[str, bool]:
    """Trim trailing comma-separated tags until all tokenizers fit max length."""
    tags = [t.strip() for t in text.split(",") if t.strip()]
    if not tags:
        return text, False

    def fits(candidate: str) -> bool:
        return all(
            _count_tokens(tok, candidate) <= _effective_max_length(tok)
            for tok in tokenizers
        )

    candidate = ", ".join(tags)
    if fits(candidate):
        return candidate, False

    while len(tags) > 1:
        tags.pop()
        candidate = ", ".join(tags)
        if fits(candidate):
            return candidate, True

    return tags[0], True


def generate(config: RunConfig, pipeline: StableDiffusionXLPipeline, progress: ProgressState) -> None:
    """Generate images in batches, saving PNGs + JSON sidecars to workspace."""
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    records = _load_prompts(config)
    # Filter out already-finalized samples
    pending = [r for r in records if not progress.is_sample_done(r["id"])]
    total = len(records)
    done = total - len(pending)
    logger.info("Resuming: %d/%d already done, %d remaining", done, total, len(pending))

    batch_num = 0
    for i in range(0, len(pending), config.batch_size):
        batch = pending[i : i + config.batch_size]
        batch_num += 1
        text_tokenizers = [
            tok for tok in [getattr(pipeline, "tokenizer", None), getattr(pipeline, "tokenizer_2", None)]
            if tok is not None
        ]

        prepared_batch = []
        for rec in batch:
            pos_prompt, pos_trimmed = _trim_prompt_to_limit(rec["positive_prompt"], text_tokenizers)
            neg_prompt, neg_trimmed = _trim_prompt_to_limit(rec["negative_prompt"], text_tokenizers)
            if pos_trimmed or neg_trimmed:
                logger.warning(
                    "Trimmed prompts for %s to fit tokenizer max length (positive_trimmed=%s, negative_trimmed=%s)",
                    rec["id"],
                    pos_trimmed,
                    neg_trimmed,
                )
            prepared_batch.append((rec, pos_prompt, neg_prompt))

        # Build generators with per-sample seeds
        generators = [
            torch.Generator(device="cpu").manual_seed(rec["seed"])
            for rec, _, _ in prepared_batch
        ]

        # Run inference
        results = pipeline(
            prompt=[pos_prompt for _, pos_prompt, _ in prepared_batch],
            negative_prompt=[neg_prompt for _, _, neg_prompt in prepared_batch],
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.resolution,
            width=config.resolution,
            generator=generators,
        )

        # Save outputs
        for (rec, pos_prompt, neg_prompt), image in zip(prepared_batch, results.images):
            sample_id = rec["id"]
            png_path = config.workspace_dir / f"{sample_id}.png"
            json_path = config.workspace_dir / f"{sample_id}.json"

            image.save(png_path)

            metadata = {
                "prompt": pos_prompt,
                "negative_prompt": neg_prompt,
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
