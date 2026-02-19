# AnimeDiffusion

1. **Prompt Generation** (`src/random_prompt/`) — produces deterministic, quality-validated prompt datasets in JSONL format
2. **Image Generation** (`src/image_gen/`) — runs Illustrious XL v2 (SDXL) inference and packages results into WebDataset shards

## Quick Start

```bash
# Install dependencies
pip install -e .

# --- Prompt Generation ---
# Pilot run (1K prompts, 1 shard)
python -m src.random_prompt --seed 42 --pilot --output-dir output/pilot

# Full run (100K prompts, 100 shards)
python -m src.random_prompt --seed 42 --output-dir output/full

# --- Image Generation ---
# Place checkpoint locally (either filename works)
# checkpoints/Illustrious-XL-v2.0.safetensors
# checkpoints/illustrious-xl-v2.safetensors

# Pilot run (10 images)
python -m src.image_gen --pilot run

# Full run (100K images)
python -m src.image_gen run

# Full run with CLI overrides
python -m src.image_gen \
  --prompts-path output/full/prompts.jsonl \
  --checkpoint-path checkpoints/illustrious-xl-v2.safetensors \
  --output-path output/images_custom \
  run
```

---

## Prompt Generation (`src/random_prompt/`)

### Features

- **Dual-channel output**: positive + negative prompts in JSONL format
- **Deterministic generation**: same seed always produces identical output
- **Shard-aware**: distributes records across shards with per-shard seeding via SHA-256
- **Quality validation**: lint rules reject contradictory traits, banned tokens, and malformed prompts
- **Diversity guarantees**: tracks unique values per feature and blocks export if thresholds aren't met
- **Near-duplicate detection**: canonicalized prompt hashing catches reordered duplicates
- **Configurable vocabulary**: all component options and weights defined in YAML

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Base random seed |
| `--count` | `100000` | Number of records |
| `--shards` | `100` | Number of shards |
| `--output-dir` | `output/full` | Output directory |
| `--pilot` | off | Pilot mode (1K records, 1 shard) |
| `--config-dir` | `configs/random_prompt/` | Config directory |

### Output

- `prompts.jsonl` — one JSON record per line with fields: `id`, `positive_prompt`, `negative_prompt`, `shard_id`, `seed`
- `manifest.json` — run metadata (seed, config hash, timestamps, record count, file hash)
- `diversity_report.json` — per-feature unique value counts and threshold results

### Architecture

```
src/random_prompt/
├── generator.py    # CLI entrypoint and orchestration
├── schema.py       # PromptComponents and PromptRecord dataclasses
├── sampler.py      # Weighted samplers, pose sampler, compatibility checks
├── assembler.py    # Prompt assembly (positive + negative)
├── lint.py         # Validation rules and rejection logic
├── diversity.py    # Diversity tracking and deduplication
├── exporter.py     # JSONL export and finalization guard
└── config.py       # Config loading from YAML
```

### Configuration

#### `configs/random_prompt/vocab.yaml`

Defines weighted option lists for each prompt component (eyes, hair, expression, etc.), pose token mappings, style modifiers, and compatibility exclusion rules.

#### `configs/random_prompt/negative_prompts.yaml`

Base negative prompt template and conditional additions for face focus, eye detail, and quality suppression.

---

## Image Generation (`src/image_gen/`)

### Model and Defaults

Image generation uses `StableDiffusionXLPipeline.from_single_file()` with Illustrious XL v2 defaults.

Default config lives at `configs/image_gen/run_config.yaml`:

| Field | Default |
|------|---------|
| `model_id` | `illustrious-xl-v2` |
| `checkpoint_path` | `checkpoints/illustrious-xl-v2.safetensors` |
| `resolution` | `1024` |
| `batch_size` | `4` |
| `num_inference_steps` | `28` |
| `guidance_scale` | `7.0` |
| `scheduler` | `EulerAncestralDiscreteScheduler` |
| `shard_size` | `1000` |

### Pipeline Stages

`python -m src.image_gen run` executes:
1. `plan` - validates prompts and builds sample manifest
2. `generate` - SDXL inference into workspace (`.png` + `.json` metadata per sample)
3. `package` - WebDataset shard writing via `webdataset.TarWriter`
4. `validate` - dataset gates (count, image integrity/resolution, metadata, ID coverage)
5. `report` - writes run summary artifacts

### CLI Overrides

The image generation CLI supports runtime overrides for key file paths:

| Flag | Description |
|------|-------------|
| `--prompts-path` | Input prompt JSONL path (overrides `prompts_path`) |
| `--checkpoint-path` | Model checkpoint path (overrides `checkpoint_path`) |
| `--output-path` | Output root directory (overrides `report_dir`, `workspace_dir`, and `shards_dir`) |

Example:

```bash
python -m src.image_gen \
  --prompts-path output/full/prompts.jsonl \
  --checkpoint-path checkpoints/illustrious-xl-v2.safetensors \
  --output-path output/images_custom \
  run
```

### Pilot Mode

`--pilot` rewrites runtime paths to `output/images_pilot/` and sets:
- `target_count = 10`
- `shard_size = 10`

### Manual Verification (Pilot)

```bash
# Run pilot
python -m src.image_gen --pilot run

# Expect 10 images and 1 shard
ls output/images_pilot/workspace/*.png | wc -l
ls output/images_pilot/shards/anime-face-*.tar | wc -l

# Check validation report
cat output/images_pilot/validation_report.json

# Check WebDataset readability
python - <<'PY'
import webdataset as wds

dataset = wds.WebDataset("output/images_pilot/shards/anime-face-000000.tar")
count = 0
for sample in dataset:
    count += 1
print("samples:", count)
PY
```

### Notes

- On lower-memory GPUs (or MPS), SDXL at `1024x1024` may require reducing `batch_size`. The runtime now auto-caps effective batch size on CUDA based on VRAM and falls back to smaller micro-batches on OOM, so `batch_size` acts as an upper bound.
- Metadata exported per sample includes: `prompt`, `negative_prompt`, `seed`, `model_id`, `resolution`, `num_inference_steps`, `guidance_scale`, `scheduler`.

---


## Testing

```bash
pytest tests/
```

Tests cover:
- **Determinism**: same seed produces identical output
- **Rules**: age style restricted to teen/adult, pose within [-15,15], all fields present, no banned tokens
- **Diversity**: threshold enforcement and finalization guard behavior
