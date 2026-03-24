"""Load a Hugging Face model with vLLM, run one prompt, and save the result to markdown."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_PROMPT = "Tell me a joke."
DEFAULT_OUTPUT_DIR = Path("data") / "vllm_outputs"


def positive_int(value: str) -> int:
    """Argparse type for integer values that must be >= 1."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


def memory_utilization(value: str) -> float:
    """Argparse type for GPU memory utilization in the range (0, 1]."""
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be a float in the range (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Set up the four CLI arguments we actually need."""
    parser = argparse.ArgumentParser(
        description="Run a single vLLM inference and save the output to markdown.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id to load. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model. Defaults to %(default)r.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Execution device. Defaults to cpu.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Explicit markdown output path. Defaults under data/vllm_outputs/.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=positive_int,
        default=1,
        help="Tensor parallel degree for multi-GPU execution. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype setting (for example: auto, float16, bfloat16). Defaults to %(default)s.",
    )
    parser.add_argument(
        "--max-model-len",
        type=positive_int,
        default=4096,
        help="Maximum context length used by vLLM. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=memory_utilization,
        default=0.9,
        help="Fraction of GPU memory vLLM may use; range (0, 1]. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Enable eager mode to reduce startup latency (can reduce inference throughput).",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization mode (free-form pass-through, e.g. awq or gptq).",
    )
    return parser


def load_vllm_components() -> tuple[type[Any], type[Any]]:
    """Import vLLM at runtime so the rest of the script stays importable without it."""
    try:
        vllm_module = importlib.import_module("vllm")
    except ImportError as exc:
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(
            "vLLM is not installed in this environment. Create a vLLM-supported Python "
            f"environment, install vLLM, and rerun this script. Current Python: {version}."
        ) from exc

    llm_cls = vllm_module.LLM
    sampling_params_cls = vllm_module.SamplingParams
    return llm_cls, sampling_params_cls


def ensure_requested_device_available(requested_device: str, detected_device: str) -> None:
    """Validate that vLLM can satisfy the requested execution device."""
    if requested_device == "auto" or requested_device == detected_device:
        return

    detected_label = detected_device or "unspecified"
    if requested_device == "cpu" and not detected_device:
        raise RuntimeError(
            "CPU execution was requested (`--device cpu`) but this vLLM runtime did not "
            "detect a usable CPU backend (detected device: unspecified). "
            "On Linux, this commonly means the installed vLLM wheel is not a CPU build. "
            "Install a CPU-compatible vLLM build, or run with `--device cuda` on a GPU machine."
        )

    raise RuntimeError(
        f"Requested device `{requested_device}` but vLLM detected `{detected_label}`. "
        "Choose a supported `--device` value for this environment."
    )


def run_inference(
    args: argparse.Namespace,
    *,
    llm_cls: type[Any] | None = None,
    sampling_params_cls: type[Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run a single inference and save the output to markdown.

    The llm_cls, sampling_params_cls, and clock parameters exist so tests can
    inject fakes without needing a real vLLM installation or GPU.
    """
    # --- Set the target device via environment variable ---
    # vLLM 0.18+ uses VLLM_TARGET_DEVICE instead of a constructor kwarg.
    # "auto" means let vLLM decide (GPU if available, else CPU).
    if args.device != "auto":
        os.environ["VLLM_TARGET_DEVICE"] = args.device

    # --- Load vLLM if no test doubles were injected ---
    if llm_cls is None or sampling_params_cls is None:
        llm_cls, sampling_params_cls = load_vllm_components()
        from vllm.platforms import current_platform

        ensure_requested_device_available(args.device, current_platform.device_type)

    now = clock() if clock is not None else datetime.now(UTC)

    # --- Build the vLLM engine config ---
    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization is not None:
        engine_kwargs["quantization"] = args.quantization

    # --- Sampling parameters (hardcoded sensible defaults) ---
    sampling_kwargs = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128}

    # --- Create engine, run inference ---
    llm = llm_cls(**engine_kwargs)
    params = sampling_params_cls(**sampling_kwargs)
    results = llm.generate([args.prompt], params)

    # vLLM returns a list of RequestOutput objects.  Each has an `outputs` list
    # of CompletionOutput objects, each with a `.text` field.  We only sent one
    # prompt, so we grab results[0].outputs[0].text.
    output_text = results[0].outputs[0].text.strip()

    # --- Determine where to save the markdown ---
    if args.output_md is not None:
        output_path = args.output_md
    else:
        # Turn "Qwen/Qwen3.5-0.8B" into "Qwen-Qwen3.5-0.8B" for a safe filename.
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", args.model).strip("-") or "model"
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"{timestamp_str}_{safe_name}.md"

    # --- Build and write the markdown report ---
    markdown = "\n".join(
        [
            "# vLLM Inference Run",
            "",
            "## Configuration",
            "",
            f"- Model: `{args.model}`",
            f"- Device: `{args.device}`",
            f"- Tensor parallel size: `{args.tensor_parallel_size}`",
            f"- Dtype: `{args.dtype}`",
            f"- Max model length: `{args.max_model_len}`",
            f"- GPU memory utilization: `{args.gpu_memory_utilization}`",
            f"- Enforce eager: `{args.enforce_eager}`",
            f"- Quantization: `{args.quantization if args.quantization is not None else 'none'}`",
            f"- Generated at: `{now.isoformat()}`",
            "",
            "## Prompt",
            "",
            "```text",
            args.prompt,
            "```",
            "",
            "## Output",
            "",
            "```text",
            output_text,
            "```",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return {
        "model": args.model,
        "prompt": args.prompt,
        "output_text": output_text,
        "device": args.device,
        "output_path": output_path,
        "timestamp": now,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_inference(args)

    print(result["output_text"])
    print(f"\nSaved markdown output to: {result['output_path']}")


if __name__ == "__main__":
    main()
