## 1. Dependencies & Scaffolding

- [x] 1.1 Add `webdataset>=0.2.100` to `pyproject.toml` dependencies
- [x] 1.2 Restore `.gitignore` from git HEAD
- [x] 1.3 Restore model-agnostic modules from git HEAD: `src/image_gen/__init__.py`, `progress.py`, `planner.py`, `report.py`

## 2. Configuration

- [x] 2.1 Create `configs/image_gen/run_config.yaml` with Illustrious XL v2 defaults (model_id, checkpoint_path, resolution=1024, batch_size=4, steps=28, guidance_scale=7.0, shard_size=1000)
- [x] 2.2 Create `src/image_gen/config.py` — restore from git HEAD and update `RunConfig` defaults: model_id=`"illustrious-xl-v2"`, checkpoint_path=`"checkpoints/illustrious-xl-v2.safetensors"`, resolution=1024, batch_size=4, num_inference_steps=28, guidance_scale=7.0

## 3. Inference Engine

- [x] 3.1 Create `src/image_gen/inference.py` — switch to `StableDiffusionXLPipeline`, load from local safetensors via `from_single_file()`, apply scheduler from config, enable xformers on CUDA
- [x] 3.2 Verify `generate()` function works with SDXL pipeline API (prompt, negative_prompt, height, width, generator, num_inference_steps, guidance_scale)

## 4. WebDataset Packaging

- [x] 4.1 Create `src/image_gen/packager.py` — use `webdataset.TarWriter` per shard, write samples with `__key__`=sample_id and `.png`/`.json` extensions, atomic writes via tmp+rename, shard-level resume

## 5. Validation

- [x] 5.1 Create `src/image_gen/validator.py` — restore from git HEAD and update `REQUIRED_METADATA_FIELDS` to include `negative_prompt` and `resolution`

## 6. CLI Entry Point

- [x] 6.1 Create `src/image_gen/__main__.py` — restore from git HEAD, add `torch.cuda.empty_cache()` after pipeline deletion in `cmd_run`

## 7. Verification

- [x] 7.1 Run `pip install -e .` to verify dependency installation
- [ ] 7.2 Run `python -m src.image_gen --pilot run` end-to-end (10 samples)
- [ ] 7.3 Verify pilot output: 10 PNGs at 1024x1024, 1 WebDataset shard, all validation gates pass
- [ ] 7.4 Verify WebDataset readability with `webdataset.WebDataset()` reader
