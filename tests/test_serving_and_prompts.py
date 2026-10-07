import pytest

from t2sbench.models.discovery import load_models_config
from t2sbench.prompts import UnverifiedPrompt, load_template, template_for
from t2sbench.serving import vllm_args

SPECS = {m["name"]: m for m in load_models_config()["models"]}


def test_precision_rules():
    a7 = vllm_args(SPECS["qwen2.5-coder-7b"])
    assert a7[a7.index("--dtype") + 1] == "bfloat16" and a7[a7.index("--kv-cache-dtype") + 1] == "fp8"
    assert a7[a7.index("--gpu-memory-utilization") + 1] == "0.92"
    a14 = vllm_args(SPECS["qwen2.5-coder-14b"])
    assert a14[1].endswith("-AWQ") and a14[a14.index("--quantization") + 1] == "awq"
    for name, spec in SPECS.items():
        if spec["backend"] == "vllm" and (spec.get("size_b") or 0) >= 22:
            assert spec.get("checkpoint"), f"{name}: 22-24B needs a ready quantized checkpoint"


def test_sqrl_serving_is_unquantized_without_reasoning_parser():
    for name in ("sqrl-4b", "sqrl-9b"):
        a = vllm_args(SPECS[name])
        assert "--quantization" not in a and "--reasoning-parser" not in a and "--kv-cache-dtype" not in a
        assert a[a.index("--max-model-len") + 1] == "16384" and a[a.index("--dtype") + 1] == "bfloat16"


def test_lora_args():
    a = vllm_args(SPECS["qwen2.5-coder-7b"], loras={"x-L1": "/a", "x-L3": "/b"})
    assert "--enable-lora" in a and a[a.index("--max-lora-rank") + 1] == "16" and "x-L1=/a" in a


def test_unverified_official_prompt_is_refused():
    with pytest.raises(UnverifiedPrompt):
        template_for(SPECS["omnisql-7b"])
    assert template_for(SPECS["omnisql-7b"], allow_unverified=True).name == "omnisql"
    assert template_for(SPECS["qwen2.5-coder-7b"]).name == "default"


def test_templates_render():
    for name in ("default", "omnisql", "arctic"):
        t = load_template(name)
        system, user = t.render("CREATE TABLE t (a INT);", "How many?", "a means x", "Example 1 ...")
        assert "CREATE TABLE t" in user and "How many?" in user and "a means x" in user and "Example 1" in user
        _, user2 = t.render("S", "Q")
        assert "Hint" not in user2 and "Examples" not in user2
