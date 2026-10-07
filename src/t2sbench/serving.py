"""vLLM command line for a configured model (consumed by scripts/serve.sh)."""

from __future__ import annotations


def vllm_args(spec: dict, port: int = 8000, loras: dict[str, str] | None = None,
              max_lora_rank: int = 16) -> list[str]:
    if spec["backend"] != "vllm":
        raise ValueError(f"{spec['name']} is not a vLLM model")
    repo = spec.get("checkpoint") or spec["hf_repo"]
    args = ["serve", repo, "--served-model-name", spec["name"], "--port", str(port),
            "--max-model-len", str(spec.get("max_model_len", 8192)),
            "--gpu-memory-utilization", str(spec.get("gpu_memory_utilization", 0.92)),
            "--seed", "42", "--generation-config", "vllm"]
    if spec.get("quantization"):
        args += ["--quantization", spec["quantization"]]
    else:
        args += ["--dtype", spec.get("dtype", "bfloat16")]
    if spec.get("kv_cache_dtype"):
        args += ["--kv-cache-dtype", spec["kv_cache_dtype"]]
    if spec.get("tool_call_parser"):
        args += ["--enable-auto-tool-choice", "--tool-call-parser", spec["tool_call_parser"]]
    # SQRL: no --reasoning-parser on purpose; we parse the raw text after </think>.
    if loras:
        args += ["--enable-lora", "--max-lora-rank", str(max_lora_rank), "--max-loras", str(min(len(loras), 8)),
                 "--lora-modules", *[f"{n}={p}" for n, p in loras.items()]]
    return args
