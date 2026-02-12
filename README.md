# AnimeDiffusion

A scalable pipeline for generating structured anime-face prompts and synthesizing images with **Waifu Diffusion v1.4**. The project consists of two stages:

1. **Prompt Generation** (`src/random_prompt/`) — produces deterministic, quality-validated prompt datasets in JSONL format
2. **Image Generation** (`src/image_gen/`) — runs batched diffusion inference and packages results into WebDataset shards

## Quick Start

```bash
# Install (includes diffusers, torch, Pillow, transformers, etc.)
pip install -e .

# --- Prompt Generation ---
# Pilot run (1K prompts, 1 shard)
python -m src.random_prompt --seed 42 --pilot --output-dir output/pilot

# Full run (100K prompts, 100 shards)
python -m src.random_prompt --seed 42 --output-dir output/full

# --- Image Generation ---
# Pilot run (1K images)
python -m src.image_gen --pilot run

# Full run (100K images)
python -m src.image_gen run
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

### Staged Pipeline

The image generation pipeline runs four stages in sequence:

1. **Plan** — loads `prompts.jsonl`, validates records, and builds a sample manifest (`sample_manifest.json`)
2. **Generate** — loads the diffusion model and runs batched inference with checkpointing; saves PNGs to the workspace directory
3. **Package** — packs workspace PNGs into WebDataset `.tar` shards
4. **Validate** — runs five completion gates on the packaged shards and writes a validation report

The `run` subcommand executes all four stages end-to-end and writes a final run report.

### CLI Options

```
python -m src.image_gen [--config PATH] [--pilot] {plan,generate,package,validate,run}
```

| Flag / Subcommand | Description |
|-------------------|-------------|
| `--config` | Path to `run_config.yaml` (default: built-in defaults) |
| `--pilot` | Pilot mode: 1000 samples into `output/images_pilot/` |
| `plan` | Build sample manifest from prompts |
| `generate` | Run batched image generation |
| `package` | Package workspace into WebDataset shards |
| `validate` | Run validation gates on packaged shards |
| `run` | Execute all stages in sequence |

### Output Artifacts

- `output/images/workspace/` — individual PNG files from inference
- `output/images/shards/` — WebDataset `.tar` shards
- `output/images/run_manifest.json` — resolved config, library versions, timestamp
- `output/images/progress.json` — resumable checkpoint state
- `output/images/validation_report.json` — gate results (pass/fail with details)
- `output/images/run_report.json` — final summary (timing, validation outcome)

### Architecture

```
src/image_gen/
├── __main__.py     # CLI entrypoint and stage orchestration
├── config.py       # RunConfig dataclass, YAML loading, and run manifest
├── planner.py      # Plan stage: load prompts, validate, build sample manifest
├── inference.py    # Generate stage: load model, batched generation with checkpointing
├── packager.py     # Package stage: write WebDataset .tar shards from workspace files
├── validator.py    # Validate stage: five completion gates for dataset integrity
├── report.py       # Run summary report generation
└── progress.py     # Progress state persistence for resumable runs
```

### Configuration

#### `configs/image_gen/run_config.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `hakurei/waifu-diffusion-v1-4` | HuggingFace model identifier |
| `prompts_path` | `output/full/prompts.jsonl` | Input prompts file |
| `workspace_dir` | `output/images/workspace` | Directory for generated PNGs |
| `shards_dir` | `output/images/shards` | Directory for WebDataset shards |
| `report_dir` | `output/images` | Directory for reports and manifests |
| `batch_size` | `8` | Images per inference batch |
| `shard_size` | `1000` | Images per `.tar` shard |
| `target_count` | `100000` | Total images to generate |
| `num_inference_steps` | `20` | Diffusion denoising steps |
| `guidance_scale` | `7.5` | Classifier-free guidance scale |
| `scheduler` | `EulerAncestralDiscreteScheduler` | Diffusion scheduler |
| `resolution` | `512` | Output image resolution (px) |

---

## Testing

```bash
pytest tests/
```

Tests cover:
- **Determinism**: same seed produces identical output
- **Rules**: age style restricted to teen/adult, pose within [-15,15], all fields present, no banned tokens
- **Diversity**: threshold enforcement and finalization guard behavior
