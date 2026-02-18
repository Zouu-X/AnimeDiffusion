## Context

The prompt generation pipeline (`src/random_prompt/`) samples numeric yaw and pitch values from a truncated normal distribution (`_truncated_normal(sigma=1.5)` in `sampler.py:43`), converts them to one of five directional text tags via `_pose_descriptor()` (`assembler.py:11`), and tracks pose diversity through `_pose_bucket()` (`diversity.py:45`). Lint validation enforces `YAW_RANGE`/`PITCH_RANGE` bounds (`lint.py:17-18`) and a semantic check prevents "closed eyes + looking direction" contradictions (`lint.py:64-78`).

Even with sigma=1.5, only ~21% of samples land in the `|yaw|<=3 AND |pitch|<=3` dead zone that produces "facing viewer". The other ~79% receive directional tags ("looking to the side", "looking away", "looking up", "looking down") that cause Illustrious XL v2 to generate side-view and profile faces undetectable by downstream face detectors, breaking the dataset pipeline.

Current token ordering in `assemble_positive` (`assembler.py:32`): quality_styles, other_styles, subject, hair_color, hair_style, eyes, face_shape, nose, mouth, expression, pose, lighting, background. Prompts use ~15-17 tags, well within the 77-token CLIP limit.

Current negative prompt (`negative_prompts.yaml`): base_negative has ~18 tags (already includes "side view, profile"), plus three conditional groups (`face_focus`, `detailed_eyes`, `quality`) totaling up to ~18 more.

## Goals / Non-Goals

**Goals:**
- Eliminate all yaw/pitch code paths and replace with constant frontal pose tags
- Add anti-side-view negative tokens for reinforced protection against non-frontal output
- Add anti-extreme-close-up negative tokens to prevent face detection breakage
- Add an `accessories` vocab category to compensate for lost pose diversity
- Update diversity tracking to replace pose buckets with accessories tracking

**Non-Goals:**
- No changes to the image generation pipeline (`src/image_gen/`)
- No changes to the export/shard format
- No CLIP token budget restructuring beyond what's needed for new tags
- No changes to the compatibility exclusion system

## Decisions

### D1: Remove yaw/pitch entirely vs. narrowing sigma further

**Choice**: Remove entirely, hardcode "frontal face, looking at viewer" as constant positive prompt tags.
**Rationale**: Even sigma=1.5 only achieves ~21% frontal. Narrowing further (e.g. sigma=0.5) would make the numeric system pointless while retaining code complexity across four modules (`sampler.py`, `assembler.py`, `diversity.py`, `lint.py`). A constant string is simpler and 100% reliable.
**Alternative considered**: sigma=0.5 — rejected because it adds no value over a constant.

### D2: Accessories as optional string field (default="")

**Choice**: Add `accessories: str = ""` to `PromptComponents` (`schema.py`), following the same pattern as `lighting` and `background`.
**Rationale**: Accessories are optional — some prompts should have none. Using `str = ""` matches the existing optional field pattern. Sampling probability is controlled by including a `""` (no-accessory) entry in the vocab with appropriate weight.
**Alternative considered**: `list[str]` like `style_modifiers` — rejected because one accessory per prompt is sufficient for diversity and simpler to lint and track.

### D3: Accessory insertion position in assembler

**Choice**: Insert accessories after `expression`, before the constant frontal pose tags (replacing the old `_pose_descriptor()` call at slot 5 in `assemble_positive`).
**Rationale**: Accessories are appearance attributes, logically grouped with face/hair/expression. The frontal pose tags act as a composition directive and belong after all appearance tokens.
**Token order becomes**: quality_styles → other_styles → subject → hair_color → hair_style → eyes → face_shape → nose → mouth → expression → accessories → "frontal face" → "looking at viewer" → lighting → background.

### D4: Anti-side-view tokens as a new conditional_negatives group

**Choice**: New `anti_side_view` conditional group in `negative_prompts.yaml`, always enabled (same unconditional pattern as `quality`).
**Rationale**: Keeps `base_negative` unchanged — it already contains "side view, profile". The new group adds reinforcing tokens ("from side", "from behind", "looking away", "looking down", "looking up", "turned head", "3/4 view") as a separate semantic unit. Easier to audit and remove later if the model changes.
**Implementation**: In `assemble_negative` (`assembler.py:85`), add unconditional inclusion of `conditional_negatives["anti_side_view"]`, identical to the existing `quality` group pattern.

### D5: Extreme close-up negatives as a separate conditional group

**Choice**: New `extreme_closeup` conditional group in `negative_prompts.yaml`, always enabled.
**Rationale**: "extreme close-up", "cropped face", "macro" are semantically distinct from the existing `face_focus` group (which handles limb/neck artifacts). A separate group keeps concerns isolated and is easier to tune independently.
**Implementation**: Unconditional inclusion in `assemble_negative`, same pattern as D4.

### D6: Accessories diversity threshold replaces pose threshold

**Choice**: Replace `"pose": 3` with `"accessories": 5` in `DEFAULT_THRESHOLDS` (`diversity.py:12`).
**Rationale**: The vocab will have ~10-12 accessory options plus the no-accessory case. Threshold of 5 ensures reasonable coverage without being hard to meet. The removed `"pose": 3` threshold tracked yaw/pitch bucket diversity which is no longer relevant.

### D7: No-accessory sampling via empty-string vocab entry

**Choice**: Add `value: "" weight: 1.0` as the first entry in the `accessories` vocab category (`vocab.yaml`).
**Rationale**: This lets the `WeightedSampler` naturally produce "no accessory" prompts without special-case code. The assembler already skips empty strings for optional fields — the `if components.lighting:` / `if components.background:` pattern applies identically. Weight 1.0 gives roughly 50% no-accessory prompts when the remaining items sum to ~1.0.

## Risks / Trade-offs

- **Reduced prompt diversity**: Removing 5 pose directions reduces variation. → Mitigated by adding ~10-12 accessory options, which provide visual diversity more relevant to face datasets.
- **Semantic lint gap**: Removing the "closed eyes + looking direction" contradiction check in `lint_semantic` (`lint.py:69-71`) leaves a gap. → Mitigated: this check is no longer needed since no looking-direction tokens will ever appear in positive prompts. The `looking_tokens` set in `lint_semantic` becomes dead code and should be removed.
- **Token budget (positive)**: Adding 2 constant frontal tags + up to 1 accessory = +3 tokens max. Current prompts use ~15-17 tags → ~18-20 max, well within the 77-token CLIP limit. → No risk.
- **Token budget (negative)**: Adding 7 anti-side-view + 3 extreme-close-up = +10 conditional tokens. Current negative has ~18 base + up to ~18 conditional = ~36. New total ~46. → Low risk, within budget.
- **Close-up weight reduction**: Reducing `close-up` style modifier weight from 0.4 to 0.2 (`vocab.yaml:287`) reduces tight framing frequency. → Acceptable trade-off: extreme close-ups are the main face detection failure mode, and `portrait` (weight 0.7) still provides face-focused framing.
