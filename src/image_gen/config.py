"""RunConfig dataclass, YAML loading, and run manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    """Configuration for an image generation run."""

    model_id: str = "hakurei/waifu-diffusion-v1-4"
    prompts_path: Path = field(default_factory=lambda: Path("output/full/prompts.jsonl"))
    workspace_dir: Path = field(default_factory=lambda: Path("output/images/workspace"))
    shards_dir: Path = field(default_factory=lambda: Path("output/images/shards"))
    report_dir: Path = field(default_factory=lambda: Path("output/images"))
    batch_size: int = 8
    shard_size: int = 1000
    target_count: int = 100_000
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    scheduler: str = "EulerAncestralDiscreteScheduler"
    resolution: int = 512
    checkpoint_path: Path = field(default_factory=lambda: Path("checkpoints/wd-1-4-anime_e2.ckpt"))

    def __post_init__(self) -> None:
        self.prompts_path = Path(self.prompts_path)
        self.workspace_dir = Path(self.workspace_dir)
        self.shards_dir = Path(self.shards_dir)
        self.report_dir = Path(self.report_dir)
        self.checkpoint_path = Path(self.checkpoint_path)


def load_config(path: Path | str | None = None) -> RunConfig:
    """Load RunConfig from a YAML file, falling back to defaults."""
    if path is None:
        return RunConfig()
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return RunConfig(**{k: v for k, v in data.items() if k in RunConfig.__dataclass_fields__})


def _config_hash(config: RunConfig) -> str:
    """Compute a short hash of the config for traceability."""
    raw = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def write_run_manifest(config: RunConfig) -> Path:
    """Write run_manifest.json with resolved config and environment info."""
    import diffusers
    import torch
    import transformers

    manifest = {
        "model_id": config.model_id,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "config_hash": _config_hash(config),
        "versions": {
            "diffusers": diffusers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_path = config.report_dir / "run_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return out_path
