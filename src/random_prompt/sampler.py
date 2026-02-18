"""Weighted samplers and compatibility checks."""

import random
from dataclasses import dataclass

from .schema import PromptComponents


@dataclass
class WeightedOption:
    value: str
    weight: float


class WeightedSampler:
    """Samples from a weighted list of options."""

    def __init__(self, options: list[dict]) -> None:
        self._options = [WeightedOption(o["value"], o["weight"]) for o in options]
        self._values = [o.value for o in self._options]
        self._weights = [o.weight for o in self._options]

    def sample(self, rng: random.Random) -> str:
        return rng.choices(self._values, weights=self._weights, k=1)[0]


class CompatibilityChecker:
    """Checks if a sampled PromptComponents has any incompatible combinations."""

    def __init__(self, exclusions: list[dict]) -> None:
        self._exclusions = exclusions

    def is_compatible(self, components: PromptComponents) -> bool:
        """Return True if no exclusion rules are violated."""
        for rule in self._exclusions:
            val_a = getattr(components, rule["component_a"], None)
            val_b = getattr(components, rule["component_b"], None)
            if val_a == rule["value_a"] and val_b == rule["value_b"]:
                return False
        return True


class ComponentSampler:
    """Orchestrates sampling of all prompt components."""

    def __init__(self, vocab: dict) -> None:
        self._samplers: dict[str, WeightedSampler] = {}
        for key in ["subject", "face_shape", "eyes", "nose", "mouth",
                     "hair_color", "hair_style", "expression", "accessories",
                     "lighting", "background"]:
            self._samplers[key] = WeightedSampler(vocab[key]["options"])

        self._style_sampler = WeightedSampler(vocab["style_modifiers"]["options"])

        exclusions = vocab.get("compatibility_exclusions", [])
        self._compat_checker = CompatibilityChecker(exclusions)

        # Number of style modifiers to sample per prompt
        self._style_count_range = (3, 6)

    def sample(self, rng: random.Random) -> PromptComponents:
        """Sample a full set of components. Returns None-compatible check separately."""
        n_styles = rng.randint(*self._style_count_range)
        style_pool = set()
        while len(style_pool) < n_styles:
            style_pool.add(self._style_sampler.sample(rng))

        return PromptComponents(
            subject=self._samplers["subject"].sample(rng),
            face_shape=self._samplers["face_shape"].sample(rng),
            eyes=self._samplers["eyes"].sample(rng),
            nose=self._samplers["nose"].sample(rng),
            mouth=self._samplers["mouth"].sample(rng),
            hair_color=self._samplers["hair_color"].sample(rng),
            hair_style=self._samplers["hair_style"].sample(rng),
            expression=self._samplers["expression"].sample(rng),
            accessories=self._samplers["accessories"].sample(rng),
            style_modifiers=sorted(style_pool),
            lighting=self._samplers["lighting"].sample(rng),
            background=self._samplers["background"].sample(rng),
        )

    def is_compatible(self, components: PromptComponents) -> bool:
        return self._compat_checker.is_compatible(components)
