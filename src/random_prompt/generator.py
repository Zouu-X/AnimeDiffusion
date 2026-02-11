"""Main entrypoint, CLI, and orchestration for random prompt generation."""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .assembler import assemble_negative, assemble_positive
from .config import (
    DEFAULT_COUNT,
    DEFAULT_SEED,
    DEFAULT_SHARDS,
    PILOT_COUNT,
    PILOT_SHARDS,
    load_negative_prompts,
    load_vocab,
)
from .diversity import DedupTracker, DiversityTracker
from .exporter import Exporter, FinalizationError
from .lint import lint_components, lint_positive_prompt, lint_record_fields
from .sampler import ComponentSampler
from .schema import PromptRecord

# Maximum resample attempts before giving up on a single record.
MAX_RESAMPLE_ATTEMPTS = 50


class RejectionTelemetry:
    """Tracks rejection counts by reason category."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def record(self, reason: str) -> None:
        self.counts[reason] += 1

    def total(self) -> int:
        return sum(self.counts.values())

    def summary(self) -> dict[str, int]:
        return dict(sorted(self.counts.items(), key=lambda x: -x[1]))


def derive_shard_seed(base_seed: int, shard_id: int) -> int:
    """Derive a deterministic per-shard seed from base seed + shard index."""
    h = hashlib.sha256(f"{base_seed}:{shard_id}".encode()).hexdigest()
    return int(h[:8], 16)


def generate_record(
    sampler: ComponentSampler,
    negative_config: dict,
    rng: random.Random,
    shard_id: int,
    record_idx: int,
    dedup: DedupTracker,
    diversity: DiversityTracker,
    telemetry: RejectionTelemetry,
) -> PromptRecord | None:
    """Generate a single valid PromptRecord, with rejection-and-regenerate loop."""
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        # Derive sub-seed for this attempt
        sub_seed = rng.randint(0, 2**31)
        sub_rng = random.Random(sub_seed)

        # Sample components
        components = sampler.sample(sub_rng)

        # Compatibility check
        if not sampler.is_compatible(components):
            telemetry.record("incompatible")
            continue

        # Lint components
        comp_violations = lint_components(components)
        if comp_violations:
            telemetry.record("lint_components")
            continue

        # Assemble prompts
        positive = assemble_positive(components)
        negative = assemble_negative(components, negative_config)

        # Lint positive prompt
        prompt_violations = lint_positive_prompt(positive)
        if prompt_violations:
            telemetry.record("lint_positive")
            continue

        # Dedup check
        if dedup.is_duplicate(positive):
            telemetry.record("duplicate")
            continue

        # Track diversity
        diversity.record(components)

        record_id = f"s{shard_id:04d}-r{record_idx:04d}"
        return PromptRecord(
            id=record_id,
            positive_prompt=positive,
            negative_prompt=negative,
            shard_id=shard_id,
            seed=sub_seed,
        )

    # Exhausted attempts
    telemetry.record("exhausted")
    return None


def generate_shard(
    shard_id: int,
    records_per_shard: int,
    base_seed: int,
    vocab: dict,
    negative_config: dict,
    dedup: DedupTracker,
    diversity: DiversityTracker,
    telemetry: RejectionTelemetry,
) -> list[PromptRecord]:
    """Generate all records for a single shard."""
    shard_seed = derive_shard_seed(base_seed, shard_id)
    rng = random.Random(shard_seed)
    sampler = ComponentSampler(vocab)

    records = []
    for i in range(records_per_shard):
        record = generate_record(
            sampler, negative_config, rng, shard_id, i,
            dedup, diversity, telemetry,
        )
        if record is not None:
            # Validate record fields
            from dataclasses import asdict
            violations = lint_record_fields(asdict(record))
            if violations:
                telemetry.record("lint_record")
                continue
            records.append(record)

    return records


def run(
    seed: int,
    count: int,
    shards: int,
    output_dir: Path,
    config_dir: Path | None = None,
) -> Path:
    """Run the full generation pipeline. Returns path to manifest."""
    # Load configs
    vocab = load_vocab(config_dir)
    negative_config = load_negative_prompts(config_dir)

    # Config hash for manifest
    config_hash = hashlib.sha256(
        json.dumps(vocab, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Shared state
    dedup = DedupTracker()
    diversity = DiversityTracker()
    telemetry = RejectionTelemetry()
    exporter = Exporter(output_dir)
    exporter.begin()

    records_per_shard = count // shards
    remainder = count % shards

    total_generated = 0
    for shard_id in range(shards):
        n = records_per_shard + (1 if shard_id < remainder else 0)
        records = generate_shard(
            shard_id, n, seed, vocab, negative_config,
            dedup, diversity, telemetry,
        )
        exporter.write_records(records)
        total_generated += len(records)

        # Progress reporting
        pct = (shard_id + 1) / shards * 100
        print(f"\r  Shard {shard_id + 1}/{shards} ({pct:.0f}%) — "
              f"{total_generated} records generated", end="", flush=True)

    print()  # newline after progress

    # Report telemetry
    if telemetry.total() > 0:
        print(f"  Rejections: {telemetry.total()} total")
        for reason, cnt in telemetry.summary().items():
            print(f"    {reason}: {cnt}")

    # Export diversity report
    diversity_report_path = output_dir / "diversity_report.json"
    report = diversity.export_report(diversity_report_path)
    print(f"  Diversity report: {diversity_report_path}")

    # Check thresholds and report
    threshold_results = diversity.check_thresholds()
    all_met = True
    for feature, (passed, unique, required) in threshold_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {feature}: {unique}/{required} unique values [{status}]")
        if not passed:
            all_met = False

    # Finalize (writes manifest, or raises if thresholds not met)
    try:
        manifest_path = exporter.finalize(
            diversity, seed, config_hash, count,
        )
        print(f"  Manifest: {manifest_path}")
        print(f"  Output: {exporter.jsonl_path}")
        print(f"  Total records: {total_generated}")
        return manifest_path
    except FinalizationError as e:
        print(f"  FINALIZATION BLOCKED: {e}", file=sys.stderr)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random anime face prompts for Waifu Diffusion v1.4"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of records to generate (overrides --pilot default)"
    )
    parser.add_argument(
        "--shards", type=int, default=None,
        help="Number of shards (overrides --pilot default)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/full"),
        help="Output directory (default: output/full)"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help=f"Pilot mode: {PILOT_COUNT} records, {PILOT_SHARDS} shard"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Config directory (default: configs/random_prompt/)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.pilot:
        count = args.count if args.count is not None else PILOT_COUNT
        shards = args.shards if args.shards is not None else PILOT_SHARDS
        if args.output_dir == Path("output/full"):
            args.output_dir = Path("output/pilot")
    else:
        count = args.count if args.count is not None else DEFAULT_COUNT
        shards = args.shards if args.shards is not None else DEFAULT_SHARDS

    print(f"Random Prompt Generator")
    print(f"  Seed: {args.seed}")
    print(f"  Count: {count}")
    print(f"  Shards: {shards}")
    print(f"  Output: {args.output_dir}")
    print()

    run(args.seed, count, shards, args.output_dir, args.config_dir)


if __name__ == "__main__":
    main()
