"""Stage 6: QLoRA fine-tuning (peft + bitsandbytes), r=16, alpha=32, dropout 0.05,
all linear layers, lr 2e-4, max 2 epochs, sequence length 4096.

Training prompts are exactly the S2 evaluation prompts of the base model (same template,
schema with values, few-shot block), the target is the gold SQL in a ```sql block.
A checkpoint is saved every 0.5 epoch and scored by execution accuracy on the VALIDATION
set (synthetic val.jsonl, or held-out BIRD *train* databases for L1); the best one becomes
adapters/<name>/final. The test sets are never touched here.

Variants: L1 = BIRD train; L2-<n> = synthetic train pool with n pairs (50/100/200/400,
nested subsets, half PL half EN); L3 = L1 adapter further trained on L2-400.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from t2sbench.datasets.synthetic.build import read_jsonl
from t2sbench.datasets.synthetic.generator import load_codes
from t2sbench.evaluate import execution_match
from t2sbench.prompts import fewshot_text, template_for
from t2sbench.schema import schema_text
from t2sbench.sqlutil import extract_sql

log = logging.getLogger(__name__)


@dataclass
class TrainHParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    epochs: float = 2.0
    max_seq_len: int = 4096
    per_device_batch: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    seed: int = 42
    val_max_new_tokens: int = 512
    val_batch: int = 4


@dataclass
class Example:
    id: str
    system: str | None
    user: str
    sql: str
    db_path: str
    lang: str = "en"


# --------------------------------------------------------------------------- data

def synthetic_subset(n: int, seed: int = 42, path: Path = Path("data/synthetic/train.jsonl")) -> list[dict]:
    """Nested subsets: the first n/2 of a fixed PL shuffle + first n/2 of a fixed EN shuffle."""
    recs = read_jsonl(path)
    rng = random.Random(seed)
    pl = sorted([r for r in recs if r["lang"] == "pl"], key=lambda r: r["id"])
    en = sorted([r for r in recs if r["lang"] == "en"], key=lambda r: r["id"])
    rng.shuffle(pl)
    rng.shuffle(en)
    return pl[: n // 2] + en[: n - n // 2]


def _render(template, dataset: str, db_path: Path, question: str, evidence: str | None, codes, fewshot) -> tuple:
    schema = schema_text(db_path, True, codes)
    return template.render(schema, question, evidence, fewshot)


def synthetic_examples(template, recs: list[dict], db: Path = Path("data/synthetic/prod.sqlite")) -> list[Example]:
    codes = load_codes()
    out = []
    for r in recs:
        ds = "syn_pl" if r["lang"] == "pl" else "syn_en"
        system, user = _render(template, ds, db, r["question"], None, codes, fewshot_text(ds))
        out.append(Example(r["id"], system, user, r["sql"], str(db), r["lang"]))
    return out


def bird_examples(template, n_val_dbs: int = 4, max_val: int = 100, seed: int = 42,
                  root: Path = Path("data/bird")) -> tuple[list[Example], list[Example]]:
    """BIRD train, without evidence (matches the bird_noev evaluation). A few train DBs are
    held out for checkpoint selection. Dev databases and questions are excluded by construction
    and asserted."""
    from t2sbench.datasets.bird import assert_train_dev_disjoint, db_path, load_questions

    train, dbdir = load_questions("train", root)
    dev, _ = load_questions("dev", root)
    assert_train_dev_disjoint(train, dev)
    dbs = sorted({q["db_id"] for q in train})
    rng = random.Random(seed)
    val_dbs = set(rng.sample(dbs, n_val_dbs))
    fs = fewshot_text("bird_noev")
    tr, va = [], []
    for q in train:
        p = db_path(dbdir, q["db_id"])
        if not p.exists():
            continue
        system, user = _render(template, "bird_noev", p, q["question"], None, None, fs)
        ex = Example(f"bird-train-{q['question_id']}", system, user, q["SQL"], str(p))
        (va if q["db_id"] in val_dbs else tr).append(ex)
    rng.shuffle(va)
    return tr, va[:max_val]


def target_text(sql: str) -> str:
    return f"```sql\n{sql.strip()}\n```"


# --------------------------------------------------------------------------- training

def _messages(ex: Example) -> list[dict]:
    return ([{"role": "system", "content": ex.system}] if ex.system else []) + [{"role": "user", "content": ex.user}]


def tokenize(tokenizer, examples: list[Example], max_len: int) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for ex in examples:
        prompt = tokenizer.apply_chat_template(_messages(ex), tokenize=False, add_generation_prompt=True)
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(target_text(ex.sql) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            dropped += 1
            continue
        rows.append({"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids})
    return rows, dropped


def evaluate_val(model, tokenizer, val: list[Example], hp: TrainHParams) -> float:
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    correct = 0
    for i in range(0, len(val), hp.val_batch):
        batch = val[i: i + hp.val_batch]
        prompts = [tokenizer.apply_chat_template(_messages(e), tokenize=False, add_generation_prompt=True) for e in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=hp.val_max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for e, seq in zip(batch, out):
            text = tokenizer.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            correct += execution_match(e.db_path, extract_sql(text), e.sql).correct
    model.train()
    tokenizer.padding_side = "right"
    return correct / max(1, len(val))


def train(base: str, name: str, train_ex: list[Example], val_ex: list[Example], out_dir: Path,
          init_adapter: Path | None = None, hp: TrainHParams | None = None,
          config_path: str = "config/models.yaml") -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainerCallback, TrainingArguments)

    from t2sbench.models.discovery import load_models_config

    hp = hp or TrainHParams()
    spec = {m["name"]: m for m in load_models_config(config_path)["models"]}[base]
    repo = spec["hf_repo"]  # original weights, quantized to 4-bit by bitsandbytes (QLoRA)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows, dropped = tokenize(tok, train_ex, hp.max_seq_len)
    log.info("%s: %d training examples (%d dropped as longer than %d tokens)", name, len(rows), dropped, hp.max_seq_len)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(repo, quantization_config=bnb, torch_dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if init_adapter:
        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=hp.r, lora_alpha=hp.alpha, lora_dropout=hp.dropout,
                                                 target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))

    steps_per_epoch = max(1, math.ceil(len(rows) / (hp.per_device_batch * hp.grad_accum)))
    save_steps = max(1, steps_per_epoch // 2)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [tok.pad_token_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

    scores: dict[str, float] = {}

    class SelectOnVal(TrainerCallback):
        def on_save(self, args, state, control, **kw):
            ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            ex = evaluate_val(kw["model"], tok, val_ex, hp)
            scores[ckpt.name] = ex
            log.info("%s %s: val EX %.4f (epoch %.2f)", name, ckpt.name, ex, state.epoch or 0)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=hp.epochs, learning_rate=hp.lr,
        per_device_train_batch_size=hp.per_device_batch, gradient_accumulation_steps=hp.grad_accum,
        warmup_ratio=hp.warmup_ratio, lr_scheduler_type="cosine", bf16=True, logging_steps=5,
        save_strategy="steps", save_steps=save_steps, save_total_limit=None, report_to=[],
        gradient_checkpointing=True, seed=hp.seed, remove_unused_columns=False, optim="paged_adamw_8bit",
    )
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=rows, data_collator=collate, callbacks=[SelectOnVal()])
    trainer.train()
    if not any(Path(args.output_dir).glob(f"checkpoint-{trainer.state.global_step}")):
        trainer.save_model(str(Path(args.output_dir) / f"checkpoint-{trainer.state.global_step}"))
        scores[f"checkpoint-{trainer.state.global_step}"] = evaluate_val(model, tok, val_ex, hp)
    train_s = time.time() - t0

    best = max(scores, key=lambda k: (scores[k], -int(k.split("-")[1])))  # ties -> earlier checkpoint
    final = out_dir / "final"
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(Path(args.output_dir) / best, final,
                    ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*", "trainer_state.json",
                                                  "training_args.bin"))
    size = sum(p.stat().st_size for p in final.rglob("*") if p.is_file() and "adapter" in p.name)
    meta = {
        "name": name, "base": base, "base_repo": repo, "init_adapter": str(init_adapter) if init_adapter else None,
        "train_examples": len(rows), "dropped_too_long": dropped, "val_examples": len(val_ex),
        "val_ex_by_checkpoint": scores, "best_checkpoint": best, "best_val_ex": scores[best],
        "steps_per_epoch": steps_per_epoch, "save_steps": save_steps, "train_seconds": train_s,
        "adapter_size_bytes": size, "hparams": asdict(hp), "quantization": "bitsandbytes nf4 4-bit (QLoRA)",
        "serving_note": "served on the stage-3 precision base model (see config/models.yaml)",
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_variant(base: str, variant: str, n: int | None = None, adapters_dir: Path = Path("adapters"),
                allow_unverified: bool = False, hp: TrainHParams | None = None) -> dict:
    from t2sbench.models.discovery import load_models_config
    from t2sbench.stages import adapter_name

    spec = {m["name"]: m for m in load_models_config()["models"]}[base]
    template = template_for(spec, allow_unverified)
    val_syn = synthetic_examples(template, read_jsonl(Path("data/synthetic/val.jsonl")))
    if variant == "L1":
        tr, va = bird_examples(template)
        return train(base, adapter_name(base, "L1"), tr, va, adapters_dir / adapter_name(base, "L1"), hp=hp)
    if variant == "L2":
        tr = synthetic_examples(template, synthetic_subset(n or 400))
        name = adapter_name(base, "L2", n or 400)
        return train(base, name, tr, val_syn, adapters_dir / name, hp=hp)
    if variant == "L3":
        init = adapters_dir / adapter_name(base, "L1") / "final"
        if not init.exists():
            raise FileNotFoundError(f"L3 needs the L1 adapter at {init}")
        tr = synthetic_examples(template, synthetic_subset(400))
        name = adapter_name(base, "L3")
        return train(base, name, tr, val_syn, adapters_dir / name, init_adapter=init, hp=hp)
    raise ValueError(variant)
