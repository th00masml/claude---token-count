"""Programmatic discovery of which configured models are actually available.

Bedrock: ``list_foundation_models`` + ``list_inference_profiles`` in the configured region;
each candidate in config/models.yaml is matched by provider and a regex on the model id,
nothing is assumed. vLLM models: the HF repo is checked for existence (and the quantized
checkpoint when one is required). Missing models are logged and skipped, never fatal.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class Availability:
    name: str
    backend: str
    role: str
    available: bool
    resolved_id: str | None
    reason: str


def load_models_config(path: str | Path = "config/models.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_bedrock(cfg: dict) -> list[Availability]:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    region = cfg["bedrock"]["region"]
    out: list[Availability] = []
    candidates = [m for m in cfg["models"] if m["backend"] == "bedrock"]
    try:
        client = boto3.client("bedrock", region_name=region)
        summaries = client.list_foundation_models()["modelSummaries"]
        try:
            profiles = client.list_inference_profiles(maxResults=1000)["inferenceProfileSummaries"]
        except (ClientError, BotoCoreError):
            profiles = []
    except (ClientError, BotoCoreError) as e:
        reason = f"bedrock unavailable in {region}: {e}"
        log.warning(reason)
        return [Availability(m["name"], "bedrock", m.get("role", "candidate"), False, None, reason)
                for m in candidates]

    for m in candidates:
        rx = re.compile(m["match"], re.IGNORECASE)
        hits = [s for s in summaries if rx.search(s["modelId"])
                and "TEXT" in s.get("outputModalities", [])
                and s.get("modelLifecycle", {}).get("status", "ACTIVE") == "ACTIVE"]
        if not hits:
            out.append(Availability(m["name"], "bedrock", m.get("role", "candidate"), False, None,
                                    f"no foundation model matching /{m['match']}/ in {region}"))
            log.warning("skip %s: not listed in %s", m["name"], region)
            continue
        hit = sorted(hits, key=lambda s: s["modelId"])[-1]
        types = hit.get("inferenceTypesSupported", [])
        if "ON_DEMAND" in types:
            resolved, reason = hit["modelId"], "on-demand"
        else:
            prof = [p for p in profiles
                    if any(hit["modelId"] in mm.get("modelArn", "") for mm in p.get("models", []))]
            if prof:
                resolved, reason = prof[0]["inferenceProfileId"], "inference profile"
            else:
                out.append(Availability(m["name"], "bedrock", m.get("role", "candidate"), False,
                                        hit["modelId"], f"listed but not invocable on demand ({types})"))
                continue
        out.append(Availability(m["name"], "bedrock", m.get("role", "candidate"), True, resolved, reason))
    return out


def discover_hf(cfg: dict) -> list[Availability]:
    import httpx

    out = []
    for m in cfg["models"]:
        if m["backend"] != "vllm":
            continue
        repo = m.get("checkpoint") or m["hf_repo"]
        if m.get("checkpoint") is None and m.get("requires_quantized_checkpoint"):
            out.append(Availability(m["name"], "vllm", m.get("role", "candidate"), False, None,
                                    "needs a ready AWQ/GPTQ 4-bit checkpoint; none configured"))
            continue
        try:
            r = httpx.get(f"https://huggingface.co/api/models/{repo}", timeout=20)
            ok = r.status_code == 200
            out.append(Availability(m["name"], "vllm", m.get("role", "candidate"), ok, repo,
                                    "found on HF" if ok else f"HF status {r.status_code}"))
        except httpx.HTTPError as e:
            out.append(Availability(m["name"], "vllm", m.get("role", "candidate"), False, repo,
                                    f"HF unreachable: {e}"))
    return out


def discover(config_path: str | Path = "config/models.yaml",
             out_path: str | Path = "results/model_availability.json") -> list[Availability]:
    cfg = load_models_config(config_path)
    res = discover_bedrock(cfg) + discover_hf(cfg)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps([asdict(a) for a in res], indent=2) + "\n")
    return res
