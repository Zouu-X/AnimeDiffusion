"""Allow running as: python -m src.image_gen"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

from .config import RunConfig, load_config, write_run_manifest
from .progress import ProgressState

logger = logging.getLogger(__name__)


def _apply_pilot(config: RunConfig) -> RunConfig:
    """Override config for a pilot (dry) run of 10 samples."""
    config.target_count = 1
    config.shard_size = 1
    config.workspace_dir = config.workspace_dir.parent.parent / "images_pilot" / "workspace"
    config.shards_dir = config.workspace_dir.parent / "shards"
    config.report_dir = config.workspace_dir.parent
    return config


def _apply_output_path(config: RunConfig, output_path: Path | str) -> RunConfig:
    """Override output directories using a single root path."""
    output_root = Path(output_path)
    config.report_dir = output_root
    config.workspace_dir = output_root / "workspace"
    config.shards_dir = output_root / "shards"
    return config


def cmd_plan(config: RunConfig) -> None:
    from .planner import plan

    plan(config)
    logger.info("Plan stage complete")


def cmd_generate(config: RunConfig) -> None:
    from .inference import generate, load_pipeline

    progress_path = config.report_dir / "progress.json"
    progress = ProgressState.load(progress_path)

    write_run_manifest(config)
    pipeline = load_pipeline(config)
    generate(config, pipeline, progress)
    logger.info("Generate stage complete")


def cmd_package(config: RunConfig) -> None:
    from .packager import package

    progress_path = config.report_dir / "progress.json"
    progress = ProgressState.load(progress_path)

    package(config, progress)
    logger.info("Package stage complete")


def cmd_validate(config: RunConfig) -> None:
    from .validator import validate

    report = validate(config)
    if report.passed:
        logger.info("Validation PASSED")
    else:
        logger.warning("Validation FAILED")
        for gate in report.gates:
            if not gate.passed:
                logger.warning("  Gate '%s': %s", gate.name, gate.details)
                for f in gate.failures[:5]:
                    logger.warning("    - %s", f)


def cmd_run(config: RunConfig) -> None:
    """Execute all stages in sequence: plan -> generate -> package -> validate."""
    from .planner import plan
    from .inference import generate, load_pipeline
    from .packager import package
    from .report import write_run_report
    from .validator import validate

    progress_path = config.report_dir / "progress.json"
    progress = ProgressState.load(progress_path)

    # Plan
    plan(config)

    # Generate
    write_run_manifest(config)
    pipeline = load_pipeline(config)
    start = time.time()
    generate(config, pipeline, progress)
    elapsed = time.time() - start

    # Free GPU memory
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Package
    package(config, progress)

    # Validate
    report = validate(config)

    # Report
    write_run_report(config, report, elapsed_seconds=elapsed)

    if report.passed:
        logger.info("Run complete: all validation gates passed")
    else:
        logger.warning("Run complete: validation FAILED — see run_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="image_gen",
        description="Anime face image generation pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to run_config.yaml (default: built-in defaults)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Pilot mode: generate 10 samples into output/images_pilot/",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output root directory (overrides report/workspace/shards dirs)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint file path (overrides checkpoint_path from config)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Build sample manifest from prompts")
    subparsers.add_parser("generate", help="Run batched image generation")
    subparsers.add_parser("package", help="Package workspace into WebDataset shards")
    subparsers.add_parser("validate", help="Run validation gates on packaged shards")
    subparsers.add_parser("run", help="Execute all stages in sequence")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()
    config = load_config(args.config)

    if args.pilot:
        config = _apply_pilot(config)
    if args.output_path is not None:
        config = _apply_output_path(config, args.output_path)
    if args.checkpoint_path is not None:
        config.checkpoint_path = Path(args.checkpoint_path)

    commands = {
        "plan": cmd_plan,
        "generate": cmd_generate,
        "package": cmd_package,
        "validate": cmd_validate,
        "run": cmd_run,
    }
    commands[args.command](config)


main()
