"""Tests for SDPA fused-only enforcement and inference runtime safeguards."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.image_gen.inference import (
    _assert_cpu_offload_disabled,
    _assert_fused_sdpa_policy,
    _compile_unet_only,
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
