"""Tests for prompt-length trimming in image generation inference."""

from src.image_gen.inference import _trim_prompt_to_limit


class _FakeTokenizer:
    def __init__(self, model_max_length: int) -> None:
        self.model_max_length = model_max_length

    def __call__(self, text: str, add_special_tokens: bool = True, truncation: bool = False):
        # Simple deterministic token estimate for testability.
        token_count = len([w for w in text.replace(",", " ").split() if w])
        if add_special_tokens:
            token_count += 2
        return {"input_ids": list(range(token_count))}


def test_trim_prompt_to_limit_no_change_when_within_budget():
    tokenizer = _FakeTokenizer(model_max_length=20)
    text = "masterpiece, 1girl, smile"

    trimmed, did_trim = _trim_prompt_to_limit(text, [tokenizer])

    assert trimmed == text
    assert did_trim is False


def test_trim_prompt_to_limit_drops_trailing_tags_until_it_fits():
    tokenizer = _FakeTokenizer(model_max_length=7)
    text = "masterpiece, best quality, 1girl, blonde hair, smile"

    trimmed, did_trim = _trim_prompt_to_limit(text, [tokenizer])

    assert did_trim is True
    assert trimmed == "masterpiece, best quality, 1girl"
