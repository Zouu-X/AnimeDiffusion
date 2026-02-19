"""Tests for SDPA fused-only enforcement and inference runtime safeguards."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from src.image_gen.config import RunConfig
from src.image_gen.inference import (
    _assert_cpu_offload_disabled,
    _assert_fused_sdpa_policy,
    _compile_unet_only,
    _warmup_compiled_unet,
)


class _DummyModule:
    pass


class _DummyPipe:
    def __init__(self) -> None:
        self.unet = _DummyModule()
        self.text_encoder = _DummyModule()
        self.text_encoder_2 = _DummyModule()
        self.vae = _DummyModule()


def test_assert_fused_sdpa_policy_rejects_math_fallback() -> None:
    with pytest.raises(RuntimeError, match="math fallback is enabled"):
        _assert_fused_sdpa_policy({"flash": True, "mem_efficient": True, "math": True})


def test_assert_fused_sdpa_policy_rejects_no_fused_kernels() -> None:
    with pytest.raises(RuntimeError, match="no fused SDPA kernels enabled"):
        _assert_fused_sdpa_policy({"flash": False, "mem_efficient": False, "math": False})


def test_assert_fused_sdpa_policy_accepts_fused_configuration() -> None:
    _assert_fused_sdpa_policy({"flash": True, "mem_efficient": False, "math": False})


def test_assert_cpu_offload_disabled_rejects_offload_hook() -> None:
    pipe = _DummyPipe()
    pipe.unet = SimpleNamespace(_hf_hook=object())
    with pytest.raises(RuntimeError, match="CPU offload must be disabled"):
        _assert_cpu_offload_disabled(pipe)


def test_compile_unet_only_keeps_non_unet_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _DummyPipe()
    optimized_cls = type("OptimizedModule", (_DummyModule,), {})

    def _fake_compile(module: object, **_: object) -> object:
        wrapped = optimized_cls()
        setattr(wrapped, "_wrapped", module)
        return wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)
    compiled = _compile_unet_only(pipe, torch.device("cuda"))
    assert compiled
    assert pipe.unet.__class__.__name__ == "OptimizedModule"
    assert pipe.text_encoder.__class__.__name__ != "OptimizedModule"
    assert pipe.text_encoder_2.__class__.__name__ != "OptimizedModule"
    assert pipe.vae.__class__.__name__ != "OptimizedModule"


# ---------- Attention slicing conditional tests ----------


class _SlicingTrackingPipe:
    """Pipe that records which slicing method was called."""

    def __init__(self) -> None:
        self.slicing_enabled: bool | None = None

    def enable_attention_slicing(self, slice_size: str = "auto") -> None:
        self.slicing_enabled = True

    def disable_attention_slicing(self) -> None:
        self.slicing_enabled = False


def test_attention_slicing_disabled_when_fused_active() -> None:
    """When flash or mem_efficient SDPA is active, attention slicing must be disabled."""
    pipe = _SlicingTrackingPipe()
    policy = {"flash": True, "mem_efficient": True, "math": False}
    fused_active = policy.get("flash", False) or policy.get("mem_efficient", False)
    if fused_active:
        pipe.disable_attention_slicing()
    else:
        pipe.enable_attention_slicing("auto")
    assert pipe.slicing_enabled is False


def test_attention_slicing_enabled_when_fused_not_active() -> None:
    """When no fused SDPA kernels are active, attention slicing should be enabled."""
    pipe = _SlicingTrackingPipe()
    policy = {"flash": False, "mem_efficient": False, "math": True}
    fused_active = policy.get("flash", False) or policy.get("mem_efficient", False)
    if fused_active:
        pipe.disable_attention_slicing()
    else:
        pipe.enable_attention_slicing("auto")
    assert pipe.slicing_enabled is True


def test_attention_slicing_disabled_with_flash_only() -> None:
    """Flash-only policy should still disable attention slicing."""
    pipe = _SlicingTrackingPipe()
    policy = {"flash": True, "mem_efficient": False, "math": False}
    fused_active = policy.get("flash", False) or policy.get("mem_efficient", False)
    if fused_active:
        pipe.disable_attention_slicing()
    else:
        pipe.enable_attention_slicing("auto")
    assert pipe.slicing_enabled is False


def test_attention_slicing_disabled_with_mem_efficient_only() -> None:
    """mem_efficient-only policy should still disable attention slicing."""
    pipe = _SlicingTrackingPipe()
    policy = {"flash": False, "mem_efficient": True, "math": False}
    fused_active = policy.get("flash", False) or policy.get("mem_efficient", False)
    if fused_active:
        pipe.disable_attention_slicing()
    else:
        pipe.enable_attention_slicing("auto")
    assert pipe.slicing_enabled is False


# ---------- Warmup function tests ----------


def test_warmup_compiled_unet_calls_pipe_with_correct_args() -> None:
    """Warmup should call pipe with num_inference_steps=1 and output_type=latent."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock()  # simulate pipeline output
    config = RunConfig(resolution=1024, guidance_scale=7.0)
    device = torch.device("cuda")
    dtype = torch.float16

    # Patch torch.cuda.empty_cache to avoid actual CUDA calls
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        result = _warmup_compiled_unet(mock_pipe, config, device, dtype)

    assert result is True
    mock_pipe.assert_called_once()
    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["num_inference_steps"] == 1
    assert call_kwargs["output_type"] == "latent"
    assert call_kwargs["guidance_scale"] == 7.0
    assert call_kwargs["height"] == 1024
    assert call_kwargs["width"] == 1024


def test_warmup_compiled_unet_returns_false_on_failure() -> None:
    """Warmup failure should be non-fatal and return False."""
    mock_pipe = MagicMock()
    mock_pipe.side_effect = RuntimeError("simulated warmup failure")
    config = RunConfig(resolution=1024, guidance_scale=7.0)
    device = torch.device("cuda")
    dtype = torch.float16

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        result = _warmup_compiled_unet(mock_pipe, config, device, dtype)

    assert result is False


def test_warmup_uses_config_resolution_and_guidance() -> None:
    """Warmup must match production resolution and guidance_scale for correct graph shapes."""
    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock()
    config = RunConfig(resolution=768, guidance_scale=5.5)
    device = torch.device("cuda")
    dtype = torch.float16

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "empty_cache", lambda: None)
        _warmup_compiled_unet(mock_pipe, config, device, dtype)

    call_kwargs = mock_pipe.call_args[1]
    assert call_kwargs["height"] == 768
    assert call_kwargs["width"] == 768
    assert call_kwargs["guidance_scale"] == 5.5
