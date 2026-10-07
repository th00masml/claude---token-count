"""Prompt templates (config/prompts/*.yaml) and few-shot examples (config/fewshot.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_DIR = Path("config/prompts")
PLACEHOLDER = "__COPY_FROM_MODEL_CARD__"


class UnverifiedPrompt(RuntimeError):
    pass


@dataclass
class PromptTemplate:
    name: str
    user: str
    system: str | None = None
    evidence_block: str = "Hint: {evidence}\n"
    examples_header: str = "Examples:\n\n"
    sampling: dict = field(default_factory=dict)
    verified: bool = False
    source: str | None = None
    card_revision: str | None = None

    def render(self, schema: str, question: str, evidence: str | None = None,
               examples: str | None = None) -> tuple[str | None, str]:
        ev = self.evidence_block.format(evidence=evidence) if evidence else ""
        ex = (self.examples_header + examples + "\n\n") if examples else ""
        user = self.user.format(schema=schema, question=question, evidence_block=ev, examples=ex)
        return (self.system.strip() if self.system else None), user.strip()


@lru_cache(maxsize=None)
def load_template(name: str, prompt_dir: str = str(PROMPT_DIR)) -> PromptTemplate:
    with open(Path(prompt_dir) / f"{name}.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    keys = PromptTemplate.__dataclass_fields__
    return PromptTemplate(**{k: v for k, v in raw.items() if k in keys})


def template_for(model_spec: dict, allow_unverified: bool = False,
                 prompt_dir: str = str(PROMPT_DIR)) -> PromptTemplate:
    t = load_template(model_spec.get("prompt_template") or "default", prompt_dir)
    if not t.verified and not allow_unverified:
        raise UnverifiedPrompt(
            f"prompt '{t.name}' for {model_spec['name']} is not verified against the model card "
            f"({t.source}); verify config/prompts/{t.name}.yaml or pass --allow-unverified-prompts")
    return t


def default_params(template: PromptTemplate):
    from t2sbench.models.base import GenParams

    s = dict(template.sampling or {})
    extra = {k: s.pop(k) for k in list(s) if k not in ("temperature", "top_p", "max_tokens", "stop")}
    return GenParams(temperature=s.get("temperature", 0.0), top_p=s.get("top_p"),
                     max_tokens=s.get("max_tokens", 2048), stop=s.get("stop"), extra=extra)


@lru_cache(maxsize=None)
def _fewshot_cfg(path: str = "config/fewshot.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fewshot_text(dataset: str, synthetic_train: list[dict] | None = None,
                 path: str = "config/fewshot.yaml") -> str:
    cfg = _fewshot_cfg(path)
    parts = []
    if dataset.startswith("bird"):
        for i, ex in enumerate(cfg["bird"], 1):
            ev = f"Hint: {ex['evidence']}\n" if dataset == "bird_ev" and ex.get("evidence") else ""
            parts.append(f"Example {i}\nDatabase schema:\n{ex['schema'].strip()}\n\n{ev}"
                         f"Question: {ex['question']}\n```sql\n{ex['sql'].strip()}\n```")
    else:
        if synthetic_train is None:
            from t2sbench.datasets.synthetic.build import read_jsonl

            synthetic_train = read_jsonl(Path("data/synthetic/train.jsonl"))
        by_id = {r["id"]: r for r in synthetic_train}
        for i, qid in enumerate(cfg["synthetic"]["ids"], 1):
            r = by_id[qid]
            parts.append(f"Example {i} (same database as below)\nQuestion: {r['question']}\n"
                         f"```sql\n{r['sql']}\n```")
    return "\n\n".join(parts)
