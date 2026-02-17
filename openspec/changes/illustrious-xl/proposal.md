## Why

The project needs to migrate its image generation pipeline from Waifu Diffusion v1.4 (SD 1.x, 512x512) to Illustrious XL v2 (SDXL-based, 1024x1024) to produce a higher-quality 100K anime face dataset. The prompt generation stage already outputs Illustrious XL v2-compatible tags (`score_9`, `score_8_up`, `source_anime`, `newest`). The old `src/image_gen/` module has been deleted and must be rebuilt for the new model architecture.

## What Changes

- **BREAKING**: Replace `StableDiffusionPipeline` with `StableDiffusionXLPipeline` for SDXL model loading
- **BREAKING**: Change default resolution from 512x512 to 1024x1024
- Load model from local `.safetensors` checkpoint via `from_single_file()` instead of HuggingFace model ID
the checkpoint will be placed in `/checkpoints/Illustrious-XL-v2.0.safetensors`
- Update default inference parameters: `batch_size=4`, `num_inference_steps=28`, `guidance_scale=7.0`
- Replace manual `tarfile` packaging with `webdataset.TarWriter` for proper WebDataset shard output
- Add `webdataset>=0.2.100` as a project dependency
- Reduce default batch size from 8 to 4 (SDXL requires ~4x more VRAM per image)

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `waifu-diffusion-inference-config`: Model changes from Waifu Diffusion v1.4 to Illustrious XL v2 (SDXL architecture), resolution from 512 to 1024, updated inference defaults
- `anime-face-generation-pipeline`: Resolution requirement changes from 512x512 to 1024x1024
- `webdataset-export`: Packaging switches from manual tarfile to `webdataset.TarWriter` library; adds `webdataset` dependency

## Impact

- **Code**: All `src/image_gen/` files must be recreated — `inference.py` (major rewrite for SDXL), `packager.py` (moderate rewrite for wds.TarWriter), `config.py` (updated defaults), `validator.py` (updated metadata fields). `planner.py`, `progress.py`, `report.py` restored unchanged.
- **Config**: `configs/image_gen/run_config.yaml` recreated with new defaults
- **Dependencies**: `webdataset>=0.2.100` added to `pyproject.toml`
- **Disk**: ~300GB workspace + ~300GB shards for 100K images at 1024x1024 PNG
- **VRAM**: Requires ~12-16GB GPU memory with batch_size=4 at 1024x1024
