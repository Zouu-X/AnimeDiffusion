## Why

Illustrious XL v2 generates side-view and profile faces when given pose descriptors like "looking away", "looking down", or "looking up". The current system samples yaw and pitch from a truncated normal distribution and converts them to directional text tags via `_pose_descriptor()`. Even after reducing sigma from 5.0 to 1.5 (commit `7a98fc7`), only ~21% of prompts receive "facing viewer" — the rest get non-frontal directions that produce faces undetectable by downstream face detectors, breaking the dataset pipeline.

## What Changes

- **BREAKING**: Remove the yaw/pitch pose system entirely — eliminate numeric sampling via `_truncated_normal()`, the `_pose_descriptor()` text conversion, `_pose_bucket()` diversity tracking, and yaw/pitch lint validation (`YAW_RANGE`/`PITCH_RANGE`).
- Fix frontal pose by hardcoding "frontal face, looking at viewer" as constant positive prompt tags on every generated prompt.
- Strengthen negative prompts against non-frontal views by adding anti-side-view tokens: "from side", "from behind", "looking away", "looking down", "looking up", "turned head", "3/4 view".
- Prevent extreme close-ups that break face detection by adding a conditional negative group for "extreme close-up", "cropped face", "macro" and reducing the `close-up` style modifier weight from 0.4 to 0.2.
- Add an `accessories` vocab category (earrings, headband, hair ribbon, choker, necklace, hairpin, hair bow, etc.) to compensate for lost pose variation and increase visual diversity. Accessories that block or cover facial features (sunglasses, masks, face coverings, goggles, blindfold, etc.) are excluded so face detection remains reliable.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `anime-face-prompt-generation`: Replace yaw/pitch pose constraints with a fixed frontal pose requirement; add accessory composition to prompt schema.
- `prompt-quality-and-diversity-constraints`: Remove pose diversity threshold (currently 8 pose variants); add accessories diversity threshold; strengthen negative prompt rules with anti-side-view and anti-extreme-close-up tokens.

## Impact

| File | Change |
|------|--------|
| `src/random_prompt/schema.py` | Remove `yaw` and `pitch` fields from `PromptComponents` |
| `src/random_prompt/sampler.py` | Remove `_truncated_normal()` and yaw/pitch sampling; add accessories sampler using weighted vocab |
| `src/random_prompt/assembler.py` | Remove `_pose_descriptor()`; replace with constant "frontal face, looking at viewer" tags; insert sampled accessories into positive prompt |
| `src/random_prompt/diversity.py` | Remove `_pose_bucket()` and pose bucket tracking; add accessories tracking |
| `src/random_prompt/lint.py` | Remove `YAW_RANGE`/`PITCH_RANGE` constants and yaw/pitch range validation; remove looking-direction semantic check |
| `configs/random_prompt/negative_prompts.yaml` | Add `anti_side_view` conditional negative group; add `extreme_closeup` conditional negative group |
| `configs/random_prompt/vocab.yaml` | Add `accessories` category with face-safe items; reduce `close-up` weight from 0.4 to 0.2 |
| `tests/test_random_prompt/test_rules.py` | Remove `test_pose_within_bounds` and `test_single_pose_direction`; add frontal-face constant tests; update `PromptComponents` constructors |
| `tests/test_random_prompt/test_diversity.py` | Remove yaw/pitch from `_make_components()`; update diversity threshold tests |

### Token Budget

- **Positive**: Current prompts use ~15-17 tags. Adding 2 fixed frontal tags plus 0-2 accessories stays well within the 77-token CLIP limit.
- **Negative**: Current base negative has ~18 tags. Adding ~7 anti-side-view tokens is manageable; low-value tokens will be audited and trimmed if needed.
