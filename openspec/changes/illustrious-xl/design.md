## Context

The image generation pipeline (`src/image_gen/`) was built for Waifu Diffusion v1.4, a Stable Diffusion 1.x model generating 512x512 images. The module has been deleted from the working directory and must be rebuilt for Illustrious XL v2, an SDXL-based model that generates at 1024x1024.

The prompt generation stage is complete — 100K prompts already exist in `output/full/prompts.jsonl` with Illustrious XL v2-compatible quality tags. The existing module architecture (plan → generate → package → validate → report) is sound and should be preserved. Most modules are model-agnostic and can be restored from git HEAD unchanged.

## Goals / Non-Goals

**Goals:**
- Rebuild `src/image_gen/` to generate 100K anime face images using Illustrious XL v2 at 1024x1024
- Load model from local `.safetensors` checkpoint via `StableDiffusionXLPipeline.from_single_file()`
- Package output in WebDataset format using `webdataset.TarWriter` with shard size 1000
- Preserve resumability, deterministic seeding, and validation gates from the original pipeline

**Non-Goals:**
- HuggingFace Hub model downloading (loading from local checkpoint only)
- Changing the prompt format or regenerating prompts
- Multi-GPU / distributed inference
- Image post-processing or filtering

## Decisions

### D1: Use `StableDiffusionXLPipeline.from_single_file()` for model loading

**Choice**: Load from local `.safetensors` file using `from_single_file()`.
**Rationale**: User has the model locally. Avoids HuggingFace authentication and download time. The `from_single_file()` API handles SDXL architecture detection automatically.
**Alternative considered**: `from_pretrained()` with HuggingFace model ID — rejected because user specified local file loading.

### D2: Use `webdataset.TarWriter` per shard (not `ShardWriter`)

**Choice**: Write each shard individually with `wds.TarWriter` using our pre-determined shard boundaries.
**Rationale**: Sample IDs are pre-assigned to shards (`s{shard_id}-r{record_idx}`). `ShardWriter` auto-splits at `maxcount` which could misalign with our shard ID scheme. `TarWriter` gives explicit control over shard boundaries while still producing spec-compliant WebDataset tars.
**Alternative considered**: `ShardWriter` with `maxcount=1000` — rejected due to shard boundary alignment risk.

### D3: Reduce default batch size to 4

**Choice**: `batch_size=4` (down from 8).
**Rationale**: SDXL at 1024x1024 uses ~4x more VRAM per image than SD 1.x at 512x512. batch_size=4 targets ~12-16GB VRAM (24GB GPU comfortable, 16GB GPU tight). Configurable via YAML for users with different hardware.

### D4: Restore model-agnostic modules from git HEAD unchanged

**Choice**: Restore `planner.py`, `progress.py`, `report.py` directly from git — no modifications.
**Rationale**: These modules operate on sample IDs, JSON records, and config values. They have no model-specific code. Changing them would be unnecessary churn.

### D5: Enable xformers memory-efficient attention when available

**Choice**: Attempt `pipe.enable_xformers_memory_efficient_attention()` on CUDA, with graceful fallback.
**Rationale**: Reduces VRAM usage significantly for SDXL. Non-fatal if xformers is not installed — inference still works, just uses more memory.

## Risks / Trade-offs

- **Disk space (~600GB total)**: 100K images at 1024x1024 PNG ~3MB each = ~300GB workspace + ~300GB in shards. → Mitigation: workspace can be cleaned after packaging; documented as expected.
- **VRAM pressure**: batch_size=4 at 1024x1024 requires ~12-16GB. → Mitigation: configurable via run_config.yaml; xformers optimization attempted automatically.
- **MPS (Apple Silicon) compatibility**: SDXL on MPS is functional but slower and memory-constrained with float16. → Mitigation: float16 is used (requires PyTorch >= 2.1, already in deps); users can reduce batch_size.
- **Checkpoint format variations**: Different Illustrious XL v2 distributions may use different safetensors layouts. → Mitigation: `from_single_file()` handles most variants; if issues arise, user can convert checkpoint format.
