"""Build Model objects from config/models.yaml + results/model_availability.json."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from t2sbench.models.budget import Budget
from t2sbench.models.cache import DiskCache
from t2sbench.models.cached import CachedModel
from t2sbench.models.discovery import load_models_config

log = logging.getLogger(__name__)


class Registry:
    def __init__(self, config_path: str | Path = "config/models.yaml",
                 availability_path: str | Path = "results/model_availability.json",
                 cache_dir: str | Path = "cache/calls", budget: Budget | None = None):
        self.cfg = load_models_config(config_path)
        self.by_name = {m["name"]: m for m in self.cfg["models"]}
        self.availability = {}
        if Path(availability_path).exists():
            self.availability = {a["name"]: a for a in json.loads(Path(availability_path).read_text())}
        self.cache = DiskCache(cache_dir)
        self.budget = budget

    def spec(self, name: str) -> dict:
        return self.by_name[name]

    def is_available(self, name: str) -> bool:
        a = self.availability.get(name)
        return bool(a and a["available"])

    def available(self, roles: tuple[str, ...] = ("candidate",), backend: str | None = None) -> list[str]:
        out = []
        for m in self.cfg["models"]:
            if m.get("role", "candidate") not in roles or (backend and m["backend"] != backend):
                continue
            if self.is_available(m["name"]):
                out.append(m["name"])
            else:
                reason = self.availability.get(m["name"], {}).get("reason", "not checked; run discover-models")
                log.warning("skipping %s: %s", m["name"], reason)
        return out

    def identity(self, name: str) -> str:
        m = self.spec(name)
        if m["backend"] == "bedrock":
            return self.availability[name]["resolved_id"]
        prec = m.get("quantization") or m.get("dtype", "auto")
        return f"{m.get('checkpoint') or m['hf_repo']}|{prec}|{m.get('max_model_len')}"

    def build(self, name: str, adapter: str | None = None, client=None):
        m = self.spec(name)
        if m["backend"] == "bedrock":
            from t2sbench.models.bedrock import BedrockModel

            inner = BedrockModel(name, self.availability[name]["resolved_id"], self.cfg["bedrock"]["region"],
                                 client=client, native_tools=m.get("tools", "native") == "native")
        else:
            from t2sbench.models.openai_compat import OpenAICompatModel

            base_url = os.environ.get("T2S_VLLM_BASE_URL") or self.cfg["vllm"]["base_url"]  # set by serve.sh
            inner = OpenAICompatModel(name, base_url, served_name=name, adapter=adapter,
                                      client=client, native_tools=m.get("tools", "native") == "native")
        return CachedModel(inner, self.cache, self.budget, identity=self.identity(name))
