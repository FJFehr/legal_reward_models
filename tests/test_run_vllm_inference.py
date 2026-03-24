from __future__ import annotations

import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_vllm_inference.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_vllm_inference", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_inference_writes_markdown(tmp_path):
    module = load_module()
    parser = module.build_parser()
    output_path = tmp_path / "result.md"
    args = parser.parse_args(
        [
            "--model",
            "Qwen/Qwen3.5-0.8B",
            "--prompt",
            "Tell me a joke.",
            "--device",
            "cpu",
            "--output-md",
            str(output_path),
        ]
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeLLM.created_with = kwargs

        def generate(self, prompts, sampling_params):
            FakeLLM.prompts = prompts
            FakeLLM.sampling_params = sampling_params
            return [SimpleNamespace(outputs=[SimpleNamespace(text="A short fake answer.")])]

    fixed_now = datetime(2026, 3, 24, 12, 0, tzinfo=UTC)
    result = module.run_inference(
        args,
        llm_cls=FakeLLM,
        sampling_params_cls=FakeSamplingParams,
        clock=lambda: fixed_now,
    )

    assert result["output_text"] == "A short fake answer."
    assert result["output_path"] == output_path
    # vLLM 0.18+ uses VLLM_TARGET_DEVICE env var, not a constructor kwarg
    assert os.environ.get("VLLM_TARGET_DEVICE") == "cpu"
    assert FakeLLM.created_with["tensor_parallel_size"] == 1
    assert FakeLLM.prompts == ["Tell me a joke."]
    assert FakeLLM.sampling_params.kwargs["max_tokens"] == 128
    assert output_path.exists()
    markdown = output_path.read_text(encoding="utf-8")
    assert "# vLLM Inference Run" in markdown
    assert "Qwen/Qwen3.5-0.8B" in markdown
    assert "Tell me a joke." in markdown
    assert "A short fake answer." in markdown


def test_load_vllm_components_raises_helpful_error(monkeypatch):
    module = load_module()

    def fake_import_module(name: str):
        if name == "vllm":
            raise ImportError("missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="vLLM is not installed"):
        module.load_vllm_components()


def test_ensure_requested_device_available_raises_on_unspecified_cpu():
    module = load_module()

    with pytest.raises(RuntimeError, match="CPU execution was requested"):
        module.ensure_requested_device_available("cpu", "")
