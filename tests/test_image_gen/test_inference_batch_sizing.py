"""Tests for runtime batch-size safety heuristics in image generation."""

from src.image_gen.inference import _is_oom_error, _recommended_max_batch_size


def test_recommended_max_batch_size_caps_10gb_sdxl_to_one():
    cap = _recommended_max_batch_size(
        resolution=1024,
        total_vram_gib=10.0,
        sdpa_fused_only_active=True,
    )
    assert cap == 1


def test_recommended_max_batch_size_allows_larger_batch_on_24gb():
    cap = _recommended_max_batch_size(
        resolution=1024,
        total_vram_gib=24.0,
        sdpa_fused_only_active=True,
    )
    assert cap == 4


def test_recommended_max_batch_size_reduces_without_sdpa_fused_only():
    cap = _recommended_max_batch_size(
        resolution=1024,
        total_vram_gib=16.0,
        sdpa_fused_only_active=False,
    )
    assert cap == 2


def test_is_oom_error_matches_cuda_oom_runtime_error():
    assert _is_oom_error(RuntimeError("CUDA out of memory"))


def test_is_oom_error_ignores_non_oom_runtime_error():
    assert not _is_oom_error(RuntimeError("some other failure"))
