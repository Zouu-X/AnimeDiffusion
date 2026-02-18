## 1. Prompt schema and sampling refactor

- [ ] 1.1 Update `src/random_prompt/schema.py` `PromptComponents` to remove `yaw`/`pitch` and add optional `accessories: str = ""`
- [ ] 1.2 Refactor `src/random_prompt/sampler.py` to remove `_truncated_normal()` and all yaw/pitch sampling paths
- [ ] 1.3 Add accessories sampling in `src/random_prompt/sampler.py` from weighted vocab entries (including empty-string no-accessory option)

## 2. Positive/negative prompt assembly changes

- [ ] 2.1 Update `src/random_prompt/assembler.py` to remove `_pose_descriptor()` and emit constant `"frontal face"` and `"looking at viewer"` tags for every positive prompt
- [ ] 2.2 Insert non-empty accessories token in `src/random_prompt/assembler.py` after `expression` and before constant frontal tags
- [ ] 2.3 Update `assemble_negative` in `src/random_prompt/assembler.py` to always include `anti_side_view` and `extreme_closeup` conditional groups

## 3. Lint and diversity rule updates

- [ ] 3.1 Remove yaw/pitch range validation (`YAW_RANGE`/`PITCH_RANGE`) and related checks from `src/random_prompt/lint.py`
- [ ] 3.2 Remove closed-eyes vs looking-direction semantic contradiction logic from `src/random_prompt/lint.py`
- [ ] 3.3 Replace pose-bucket tracking with accessories tracking in `src/random_prompt/diversity.py` and update default thresholds to `accessories: 5`

## 4. Configuration and vocab updates

- [ ] 4.1 Add `anti_side_view` and `extreme_closeup` groups to `configs/random_prompt/negative_prompts.yaml` with the specified anti-side/anti-close-up tokens
- [ ] 4.2 Add `accessories` category to `configs/random_prompt/vocab.yaml` with face-safe items and a `value: ""` no-accessory entry with positive weight
- [ ] 4.3 Reduce `close-up` style modifier weight in `configs/random_prompt/vocab.yaml` from `0.4` to `0.2`

## 5. Test updates

- [ ] 5.1 Update `tests/test_random_prompt/test_rules.py` constructors/usages to match new `PromptComponents` fields and remove yaw/pitch assumptions
- [ ] 5.2 Add/adjust `tests/test_random_prompt/test_rules.py` assertions for constant frontal tags and unconditional anti-side-view/extreme-closeup negatives
- [ ] 5.3 Update `tests/test_random_prompt/test_diversity.py` fixtures and threshold expectations to use accessories diversity instead of pose diversity

## 6. Verification

- [ ] 6.1 Run `pytest tests/test_random_prompt/test_rules.py tests/test_random_prompt/test_diversity.py`
- [ ] 6.2 Run a deterministic prompt-generation smoke check (e.g., `python -m src.random_prompt`) to confirm prompts no longer contain directional pose tags and always include frontal tags
