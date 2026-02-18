"""Diversity tracking, deduplication, and reporting."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .schema import PromptComponents


# Minimum unique values required per feature.
DEFAULT_THRESHOLDS = {
    "eyes": 20,
    "hair_color": 10,
    "hair_style": 10,
    "expression": 8,
    "accessories": 5,
}


class DiversityTracker:
    """Tracks unique values seen across generated prompts."""

    def __init__(self, thresholds: dict[str, int] | None = None) -> None:
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, components: PromptComponents) -> None:
        """Record component values for diversity tracking."""
        self._record_value("eyes", components.eyes)
        self._record_value("hair_color", components.hair_color)
        self._record_value("hair_style", components.hair_style)
        self._record_value("expression", components.expression)
        if components.accessories:
            self._record_value("accessories", components.accessories)

    def _record_value(self, feature: str, value: str) -> None:
        self._seen[feature].add(value)
        self._counts[feature][value] += 1

    def check_thresholds(self) -> dict[str, tuple[bool, int, int]]:
        """Check diversity thresholds.
        Returns {feature: (passed, unique_count, required_count)}."""
        results = {}
        for feature, required in self._thresholds.items():
            unique = len(self._seen.get(feature, set()))
            results[feature] = (unique >= required, unique, required)
        return results

    def all_thresholds_met(self) -> bool:
        """Return True if all diversity thresholds are met."""
        return all(passed for passed, _, _ in self.check_thresholds().values())

    def export_report(self, output_path: Path) -> dict:
        """Export diversity report as JSON."""
        report = {
            "thresholds": self._thresholds,
            "unique_counts": {
                feature: len(values) for feature, values in self._seen.items()
            },
            "threshold_results": {},
            "per_feature_distribution": {},
        }
        for feature, (passed, unique, required) in self.check_thresholds().items():
            report["threshold_results"][feature] = {
                "passed": passed,
                "unique": unique,
                "required": required,
            }
        for feature, counts in self._counts.items():
            report["per_feature_distribution"][feature] = dict(
                sorted(counts.items(), key=lambda x: -x[1])
            )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        return report


class DedupTracker:
    """Near-duplicate detection via canonicalized prompt hashing."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def is_duplicate(self, positive_prompt: str) -> bool:
        """Check if a canonicalized version of this prompt has been seen."""
        canonical = self._canonicalize(positive_prompt)
        h = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    @staticmethod
    def _canonicalize(prompt: str) -> str:
        """Normalize prompt for dedup: lowercase, sort tokens, strip whitespace."""
        tokens = sorted(t.strip().lower() for t in prompt.split(",") if t.strip())
        return "|".join(tokens)
