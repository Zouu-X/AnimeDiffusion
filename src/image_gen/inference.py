"""Inference stage: load model, batched image generation with checkpointing."""

from __future__ import annotations

import gc
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


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if an exception looks like a CUDA out-of-memory failure."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def _recommended_max_batch_size(
    resolution: int,
    total_vram_gib: float,
    xformers_enabled: bool,
) -> int:
    """Heuristic cap that avoids spilling SDXL generation into shared GPU memory."""
    if resolution >= 1024:
        if total_vram_gib < 11:
            cap = 1
        elif total_vram_gib < 14:
            cap = 2
        elif total_vram_gib < 20:
            cap = 3
        else:
            cap = 4
    elif resolution >= 768:
        if total_vram_gib < 8:
            cap = 1
        elif total_vram_gib < 12:
            cap = 2
        elif total_vram_gib < 20:
            cap = 4
        else:
            cap = 6
    else:
        if total_vram_gib < 6:
            cap = 1
        elif total_vram_gib < 8:
            cap = 2
        elif total_vram_gib < 12:
            cap = 4
        else:
            cap = 8
    if not xformers_enabled:
        cap = max(1, cap - 1)
    return cap


def _resolve_runtime_batch_size(
    config: RunConfig,
    pipeline: StableDiffusionXLPipeline,
) -> int:
    """Resolve a safe effective batch size for this runtime/device."""
    requested = max(1, config.batch_size)
    if not torch.cuda.is_available():
        return requested

    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_vram_gib = props.total_memory / (1024 ** 3)
    except Exception:
        return requested

    xformers_enabled = bool(getattr(pipeline, "_xformers_enabled", False))
    recommended = _recommended_max_batch_size(
        resolution=config.resolution,
        total_vram_gib=total_vram_gib,
        xformers_enabled=xformers_enabled,
    )
    effective = min(requested, recommended)
    if effective < requested:
        logger.warning(
            (
                "Capping batch_size from %d to %d for %.1f GiB VRAM at %dx%d "
                "(xformers=%s) to avoid shared-memory fallback."
            ),
            requested,
            effective,
            total_vram_gib,
            config.resolution,
            config.resolution,
            xformers_enabled,
        )
    return max(1, effective)


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
    xformers_enabled = False
    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            xformers_enabled = True
            logger.info("Enabled xformers memory efficient attention")
        except Exception as exc:
            logger.warning("xformers not available, continuing without it: %s", exc)
        try:
            pipe.enable_attention_slicing("auto")
            logger.info("Enabled attention slicing")
        except Exception as exc:
            logger.warning("Could not enable attention slicing: %s", exc)
        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        except Exception as exc:
            logger.warning("Could not enable VAE slicing: %s", exc)
        try:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        except Exception as exc:
            logger.warning("Could not enable VAE tiling: %s", exc)
    setattr(pipe, "_xformers_enabled", xformers_enabled)

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
    try:
        encoded = tokenizer(  # type: ignore[misc]
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    except TypeError:
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
    runtime_batch_size = _resolve_runtime_batch_size(config, pipeline)
    logger.info("Resuming: %d/%d already done, %d remaining", done, total, len(pending))

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
    while i < len(pending):
        batch = pending[i : i + runtime_batch_size]
        batch_num += 1

        prepared_batch = []
        for rec in batch:
            pos_before = [_count_tokens(tok, rec["positive_prompt"]) for tok in text_tokenizers]
            neg_before = [_count_tokens(tok, rec["negative_prompt"]) for tok in text_tokenizers]
            pos_prompt, pos_trimmed = _trim_prompt_to_limit(rec["positive_prompt"], text_tokenizers)
            neg_prompt, neg_trimmed = _trim_prompt_to_limit(rec["negative_prompt"], text_tokenizers)
            pos_after = [_count_tokens(tok, pos_prompt) for tok in text_tokenizers]
            neg_after = [_count_tokens(tok, neg_prompt) for tok in text_tokenizers]
            if pos_trimmed or neg_trimmed:
                logger.warning(
                    (
                        "Trimmed prompts for %s to fit tokenizer max length "
                        "(positive_trimmed=%s %s->%s, negative_trimmed=%s %s->%s)"
                    ),
                    rec["id"],
                    pos_trimmed,
                    max(pos_before) if pos_before else 0,
                    max(pos_after) if pos_after else 0,
                    neg_trimmed,
                    max(neg_before) if neg_before else 0,
                    max(neg_after) if neg_after else 0,
                )
            prepared_batch.append((rec, pos_prompt, neg_prompt))

        micro_batch_size = len(prepared_batch)
        j = 0
        while j < len(prepared_batch):
            micro_batch = prepared_batch[j : j + micro_batch_size]
            generators = [
                torch.Generator(device="cpu").manual_seed(rec["seed"])
                for rec, _, _ in micro_batch
            ]

            try:
                results = pipeline(
                    prompt=[pos_prompt for _, pos_prompt, _ in micro_batch],
                    negative_prompt=[neg_prompt for _, _, neg_prompt in micro_batch],
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
                runtime_batch_size = min(runtime_batch_size, reduced)
                gc.collect()
                torch.cuda.empty_cache()
                continue

            for (rec, pos_prompt, neg_prompt), image in zip(micro_batch, results.images):
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

            done += len(micro_batch)
            progress.save()
            j += len(micro_batch)
            del results

        i += len(batch)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Batch %d: generated %d/%d samples (runtime_batch_size=%d)",
            batch_num,
            done,
            total,
            runtime_batch_size,
        )

    logger.info("Generation complete: %d samples", total)
