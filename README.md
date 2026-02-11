# AnimeDiffusion — Random Prompt Generator

A scalable random prompt generator producing structured anime-face prompts for **Waifu Diffusion v1.4** image synthesis.

## Features

- **Dual-channel output**: positive + negative prompts in JSONL format
- **Deterministic generation**: same seed always produces identical output
- **Shard-aware**: distributes records across shards with per-shard seeding via SHA-256
- **Quality validation**: lint rules reject contradictory traits, banned tokens, and malformed prompts
- **Diversity guarantees**: tracks unique values per feature and blocks export if thresholds aren't met
- **Near-duplicate detection**: canonicalized prompt hashing catches reordered duplicates
- **Configurable vocabulary**: all component options and weights defined in YAML

## Quick Start

```bash
# Install dependencies
pip install pyyaml pytest

# Pilot run (1K prompts, 1 shard)
python -m src.random_prompt --seed 42 --pilot --output-dir output/pilot

# Full run (100K prompts, 100 shards)
python -m src.random_prompt --seed 42 --output-dir output/full

# Custom count
python -m src.random_prompt --seed 42 --count 5000 --shards 10 --output-dir output/custom
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Base random seed |
| `--count` | `100000` | Number of records |
| `--shards` | `100` | Number of shards |
| `--output-dir` | `output/full` | Output directory |
| `--pilot` | off | Pilot mode (1K records, 1 shard) |
| `--config-dir` | `configs/random_prompt/` | Config directory |

## Output

- `prompts.jsonl` — one JSON record per line with fields: `id`, `positive_prompt`, `negative_prompt`, `shard_id`, `seed`
- `manifest.json` — run metadata (seed, config hash, timestamps, record count, file hash)
- `diversity_report.json` — per-feature unique value counts and threshold results

## Architecture

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

## Configuration

### `configs/random_prompt/vocab.yaml`

Defines weighted option lists for each prompt component (eyes, hair, expression, etc.), pose token mappings, style modifiers, and compatibility exclusion rules.

### `configs/random_prompt/negative_prompts.yaml`

Base negative prompt template and conditional additions for face focus, eye detail, and quality suppression.

## Testing

```bash
pytest tests/
```

Tests cover:
- **Determinism**: same seed produces identical output
- **Rules**: age style restricted to teen/adult, pose within [-15,15], all fields present, no banned tokens
- **Diversity**: threshold enforcement and finalization guard behavior
