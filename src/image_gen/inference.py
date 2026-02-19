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

# Optimization path is tuned for Databricks A10G runtime used in production.
DATABRICKS_A10G_ASSUMPTIONS = {
    "torch": "2.3.1+cu121",
    "diffusers": "0.36.0",
}


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
    sdpa_fused_only_active: bool,
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
    if not sdpa_fused_only_active:
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

    sdpa_fused_only_active = bool(getattr(pipeline, "_sdpa_fused_only_active", False))
    recommended = _recommended_max_batch_size(
        resolution=config.resolution,
        total_vram_gib=total_vram_gib,
        sdpa_fused_only_active=sdpa_fused_only_active,
    )
    effective = min(requested, recommended)
    if effective < requested:
        logger.warning(
            (
                "Capping batch_size from %d to %d for %.1f GiB VRAM at %dx%d "
                "(sdpa_fused_only=%s) to avoid shared-memory fallback."
            ),
            requested,
            effective,
            total_vram_gib,
            config.resolution,
            config.resolution,
            sdpa_fused_only_active,
        )
    return max(1, effective)


def _sdpa_kernel_policy() -> dict[str, bool]:
    """Return active SDPA kernel policy flags from torch CUDA backends."""
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        raise RuntimeError("CUDA backends are unavailable; SDPA fused-only mode requires CUDA.")

    def _read_flag(name: str) -> bool:
        getter = getattr(cuda_backend, name, None)
        if getter is None:
            return False
        return bool(getter())

    return {
        "flash": _read_flag("flash_sdp_enabled"),
        "mem_efficient": _read_flag("mem_efficient_sdp_enabled"),
        "math": _read_flag("math_sdp_enabled"),
    }


def _assert_fused_sdpa_policy(policy: dict[str, bool]) -> None:
    """Validate fused-only SDPA policy and raise clear errors when invalid."""
    if policy["math"]:
        raise RuntimeError(
            "Invalid SDPA policy: math fallback is enabled; fused-only enforcement requires math=False."
        )
    if not (policy["flash"] or policy["mem_efficient"]):
        raise RuntimeError(
            "Invalid SDPA policy: no fused SDPA kernels enabled; expected flash and/or mem-efficient."
        )


def _configure_fused_sdpa_policy() -> dict[str, bool]:
    """Enable fused SDPA kernels and disable math fallback."""
    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is None:
        raise RuntimeError("CUDA backends are unavailable; cannot configure SDPA fused-only mode.")
    for setter_name, enabled in (
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", False),
    ):
        setter = getattr(cuda_backend, setter_name, None)
        if setter is None:
            raise RuntimeError(f"Missing torch CUDA SDPA control: {setter_name}")
        setter(enabled)

    policy = _sdpa_kernel_policy()
    _assert_fused_sdpa_policy(policy)
    return policy


def _configure_sdpa_attention_backend(pipe: StableDiffusionXLPipeline) -> None:
    """Force Diffusers attention processors onto the PyTorch SDPA path."""
    try:
        pipe.unet.set_default_attn_processor()
    except Exception as exc:
        raise RuntimeError(
            "Failed to configure UNet for default SDPA attention processors."
        ) from exc
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "set_default_attn_processor"):
        try:
            vae.set_default_attn_processor()
        except Exception as exc:
            raise RuntimeError(
                "Failed to configure VAE for default SDPA attention processors."
            ) from exc


def _validate_fused_sdpa_runtime(device: torch.device, dtype: torch.dtype) -> None:
    """Fail fast if fused-only SDPA cannot execute in the active runtime."""
    q = torch.randn((1, 8, 64, 64), device=device, dtype=dtype)
    try:
        _ = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            "Fused SDPA preflight failed; fused-kernel acceleration is unavailable in this runtime."
        ) from exc


def _is_compiled_module(module: object | None) -> bool:
    if module is None:
        return False
    return module.__class__.__name__ == "OptimizedModule"


def _compile_unet_only(pipe: StableDiffusionXLPipeline, device: torch.device) -> bool:
    """Compile only UNet for inference acceleration."""
    if device.type != "cuda":
        return False
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available; UNet compile cannot be enabled.")
    pipe.unet = torch.compile(
        pipe.unet,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=False,
    )
    if not _is_compiled_module(pipe.unet):
        raise RuntimeError("UNet compile verification failed; expected compiled UNet module.")
    for name in ("text_encoder", "text_encoder_2", "vae"):
        if _is_compiled_module(getattr(pipe, name, None)):
            raise RuntimeError(f"Only UNet may be compiled, but {name} appears compiled.")
    if _is_compiled_module(getattr(pipe, "scheduler", None)):
        raise RuntimeError("Only UNet may be compiled, but scheduler appears compiled.")
    return True


def _assert_cpu_offload_disabled(pipe: StableDiffusionXLPipeline) -> None:
    """Ensure no CPU offload hooks are active for this optimization path."""
    offload_hooks: list[str] = []
    for name in ("unet", "text_encoder", "text_encoder_2", "vae"):
        module = getattr(pipe, name, None)
        if module is None:
            continue
        if getattr(module, "_hf_hook", None) is not None:
            offload_hooks.append(name)
    if offload_hooks:
        joined = ", ".join(sorted(offload_hooks))
        raise RuntimeError(
            f"CPU offload must be disabled for SDPA optimization, found offload hooks on: {joined}"
        )


def _log_runtime_acceleration_state(
    *,
    device: torch.device,
    policy: dict[str, bool],
    unet_compiled: bool,
) -> None:
    """Emit structured startup state for acceleration debugging."""
    logger.info(
        "inference_runtime_state=%s",
        json.dumps(
            {
                "attention_backend": "sdpa",
                "device": device.type,
                "sdpa_policy": policy,
                "unet_compiled": unet_compiled,
                "assumptions": DATABRICKS_A10G_ASSUMPTIONS,
            },
            sort_keys=True,
        ),
    )


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
    _configure_sdpa_attention_backend(pipe)
    unet_compiled = False
    sdpa_policy = {"flash": False, "mem_efficient": False, "math": True}
    if device.type == "cuda":
        sdpa_policy = _configure_fused_sdpa_policy()
        _validate_fused_sdpa_runtime(device, dtype)
        unet_compiled = _compile_unet_only(pipe, device)
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
    _assert_cpu_offload_disabled(pipe)
    setattr(pipe, "_attention_backend", "sdpa")
    setattr(pipe, "_sdpa_fused_only_active", bool(device.type == "cuda"))
    setattr(pipe, "_unet_compiled", unet_compiled)
    _log_runtime_acceleration_state(
        device=device,
        policy=sdpa_policy,
        unet_compiled=unet_compiled,
    )

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
