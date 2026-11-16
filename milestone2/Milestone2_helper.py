"""Helper code for Milestone 2: Qwen3-1.7B on Hungarian SIB-200 and Belebele.

Everything an experiment needs lives here, so the notebook (Milestone2.ipynb) stays short:

* data loading      SIB-200 (from the original repo) and Belebele (Hugging Face)
* prompts           the model's chat template with an English instruction
* scoring           log-likelihood of every possible answer (no free-text parsing needed),
                    plus greedy generation to measure format-following of the untrained model
* LoRA training     Unsloth FastLanguageModel + LoRA/QLoRA, loss only on the answer tokens
* run logging       every run (also failed ones) appends one row to results/runs.csv

Each experiment runs in its own subprocess (`launch(cfg)`) so that GPU memory / time measurements
start from zero and an out-of-memory crash is recorded instead of killing the notebook:

    python m2_helper.py --config '{"run_id": "...", "kind": "lora", ...}'
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SIB_DIR = REPO_ROOT / "external" / "sib-200"
RESULTS_DIR = HERE / "results"
PRED_DIR = RESULTS_DIR / "predictions"
LOG_DIR = RESULTS_DIR / "logs"
RUNS_CSV = RESULTS_DIR / "runs.csv"
SPLIT_FILE = RESULTS_DIR / "belebele_split.json"

SIB_LABELS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
LETTERS = ["A", "B", "C", "D"]
LANGS = {"hun": "hun_Latn", "eng": "eng_Latn"}

# Column order of runs.csv (new keys are appended automatically)
RUN_COLUMNS = [
    "run_id", "group", "status", "kind", "model", "load_in_4bit", "train_data",
    "seed", "r", "lora_alpha", "target_modules", "lr", "epochs", "batch_size", "grad_accum",
    "trainable_params", "total_params", "trainable_pct", "train_time_s", "train_steps",
    "peak_mem_load_gb", "peak_mem_train_gb", "peak_mem_alloc_train_gb",
    "sib_hun_dev_acc", "sib_hun_test_acc", "sib_hun_test_f1",
    "sib_eng_test_acc", "sib_eng_test_f1",
    "belebele_hun_heldout_acc", "belebele_eng_heldout_acc", "belebele_hun_full_acc", "belebele_eng_full_acc",
    "sib_hun_test_gen_acc", "sib_hun_test_gen_valid", "belebele_hun_full_gen_acc", "belebele_hun_full_gen_valid",
    "eval_time_s", "gpu", "backend", "versions", "timestamp", "notes", "error",
]
CFG_KEYS = ["run_id", "group", "kind", "model", "load_in_4bit", "train_data", "seed", "r", "lora_alpha",
            "lr", "epochs", "batch_size", "grad_accum", "notes"]


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def ensure_sib200() -> Path:
    if not SIB_DIR.exists():
        SIB_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/dadelani/sib-200.git", str(SIB_DIR)], check=True)
    return SIB_DIR


def load_sib(lang: str, split: str) -> pd.DataFrame:
    """split in {train, dev, test}; columns index_id, category, text."""
    ensure_sib200()
    return pd.read_csv(SIB_DIR / "data" / "annotated" / lang / f"{split}.tsv", sep="\t")


def load_belebele(lang: str) -> pd.DataFrame:
    from datasets import load_dataset
    df = load_dataset("facebook/belebele", lang, split="test").to_pandas()
    df["qid"] = df["link"] + "#" + df["question_number"].astype(str)
    df["gold"] = df["correct_answer_num"].astype(int) - 1
    return df


def belebele_split(seed: int = 0, train_frac: float = 2 / 3) -> dict:
    """Fixed passage-level split of Belebele, the same in every language (saved to
    results/belebele_split.json). We evaluate on the 'heldout' part: 299 questions on 1/3 of the passages."""
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    links = sorted(load_belebele("eng_Latn")["link"].unique())
    random.Random(seed).shuffle(links)
    n_train = round(len(links) * train_frac)
    split = {"seed": seed, "train_links": sorted(links[:n_train]), "heldout_links": sorted(links[n_train:])}
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    return split


def belebele_part(lang: str, part: str) -> pd.DataFrame:
    """part = 'full' (900 questions) or 'heldout' (299 questions)."""
    df = load_belebele(lang)
    if part == "full":
        return df.reset_index(drop=True)
    links = set(belebele_split()[f"{part}_links"])
    return df[df["link"].isin(links)].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------------
SIB_INSTR = ("Classify the topic of the following sentence. Answer with exactly one of these labels: "
             + ", ".join(SIB_LABELS) + ".\n\nSentence: {text}")
BELE_INSTR = ("Read the passage and answer the question. Answer with the letter of the correct option only "
              "(A, B, C or D).\n\nPassage: {passage}\n\nQuestion: {question}\nA. {a1}\nB. {a2}\nC. {a3}\nD. {a4}")


def build_examples(task: str, df: pd.DataFrame) -> list[dict]:
    """Unified format: {'id', 'user', 'answer', 'options', 'gold'}."""
    out = []
    if task == "sib":
        for _, r in df.iterrows():
            out.append({"id": r["index_id"], "user": SIB_INSTR.format(text=r["text"]), "answer": r["category"],
                        "options": SIB_LABELS, "gold": SIB_LABELS.index(r["category"])})
    elif task == "belebele":
        for _, r in df.iterrows():
            user = BELE_INSTR.format(passage=r["flores_passage"], question=r["question"], a1=r["mc_answer1"],
                                     a2=r["mc_answer2"], a3=r["mc_answer3"], a4=r["mc_answer4"])
            out.append({"id": r["qid"], "user": user, "answer": LETTERS[r["gold"]],
                        "options": LETTERS, "gold": int(r["gold"])})
    else:
        raise ValueError(task)
    return out


def chat_kwargs(tokenizer) -> dict:
    """Qwen3: disable the 'thinking' block so the model answers directly."""
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    return {"enable_thinking": False} if "enable_thinking" in tmpl else {}


def render_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True, **chat_kwargs(tokenizer))


def render_completion(tokenizer, user: str, answer: str) -> tuple[str, str]:
    """Returns (prompt_text, completion_text) such that prompt+completion is the full chat incl. the
    end-of-turn token. The completion is exactly what the model is trained / scored on."""
    prompt = render_prompt(tokenizer, user)
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    full = tokenizer.apply_chat_template(msgs, tokenize=False, **chat_kwargs(tokenizer))
    completion = full[len(prompt):] if full.startswith(prompt) else answer + (tokenizer.eos_token or "")
    return prompt, completion.rstrip("\n")  # trailing newline after end-of-turn is not part of the answer


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def score_options(model, tokenizer, examples: list[dict], batch_tokens: int = 6000) -> np.ndarray:
    """Return an (n_examples, n_options) array of log-likelihoods.

    If every option is a single token (A/B/C/D): log p(option | prompt), one forward pass per example.
    Otherwise (topic labels): sum of log p(option tokens + end-of-turn | prompt), one sequence per option.
    """
    import torch
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    opts = examples[0]["options"]
    first_token = all(len(_encode(tokenizer, o)) == 1 for o in opts)

    seqs = []  # (example_idx, option_idx, ids, n_prompt)
    for i, e in enumerate(examples):
        if first_token:
            p_ids = _encode(tokenizer, render_prompt(tokenizer, e["user"]))
            seqs.append((i, -1, p_ids, len(p_ids)))
        else:
            for j, o in enumerate(e["options"]):
                p, c = render_completion(tokenizer, e["user"], o)
                p_ids, c_ids = _encode(tokenizer, p), _encode(tokenizer, c)
                seqs.append((i, j, p_ids + c_ids, len(p_ids)))

    first_ids = [_encode(tokenizer, o)[0] for o in opts]
    scores = np.full((len(examples), len(opts)), -np.inf, dtype=np.float64)
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
    base = model.get_base_model() if hasattr(model, "get_base_model") else model  # unwrap LoRA
    inner, head = base.model, base.get_output_embeddings()
    model.eval()
    k = 0
    with torch.no_grad():
        while k < len(order):
            batch = []
            while k < len(order):  # sorted by length -> the newest item is the longest
                s = seqs[order[k]]
                if batch and (len(batch) + 1) * len(s[2]) > batch_tokens:
                    break
                batch.append(s)
                k += 1
            L = max(len(s[2]) for s in batch)
            ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for b, s in enumerate(batch):  # right padding: real tokens never attend to padding
                ids[b, :len(s[2])] = torch.tensor(s[2])
                att[b, :len(s[2])] = 1
            hidden = inner(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
            # Only the positions we need go through the LM head (logits for every position would
            # not fit in 6 GB of GPU memory).
            for b, (i, j, s_ids, n_p) in enumerate(batch):
                if first_token:
                    logp = torch.log_softmax(head(hidden[b, n_p - 1]).float(), dim=-1)
                    scores[i] = [logp[t].item() for t in first_ids]
                else:
                    pos = torch.arange(n_p - 1, len(s_ids) - 1, device=device)
                    logp = torch.log_softmax(head(hidden[b, pos]).float(), dim=-1)
                    tgt = torch.tensor(s_ids[n_p:], device=device)
                    scores[i, j] = logp.gather(1, tgt[:, None]).sum().item()
    return scores


def generate_answers(model, tokenizer, examples, max_new_tokens: int = 12, batch_size: int = 16):
    """Greedy generation (left padding). Used only to measure format-following of the untrained model."""
    import torch
    device = next(model.parameters()).device
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompts = [render_prompt(tokenizer, e["user"]) for e in examples]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                res[i] = t
    tokenizer.padding_side = old_side
    return res


def parse_generation(text: str, options: list[str]) -> tuple[int | None, bool]:
    """Returns (answer, exact_format).
    exact_format: the stripped output is exactly one option (case-insensitive, trailing '.' allowed).
    answer: lenient parse = the option mentioned first in the output (letters must stand alone)."""
    t = text.strip().strip(".*").strip().lower()
    for j, o in enumerate(options):
        if t == o.lower():
            return j, True
    hits = []
    for j, o in enumerate(options):
        m = re.search(r"\b" + re.escape(o) + r"\b", text) if len(o) == 1 else re.search(re.escape(o.lower()), text.lower())
        if m:
            hits.append((m.start(), j))
    return (min(hits)[1] if hits else None), False


def evaluate(model, tokenizer, task, lang, split, run_id=None, with_generation=False):
    """Score one evaluation set. Returns dict of metrics and saves per-example predictions."""
    from sklearn.metrics import accuracy_score, f1_score
    df = load_sib(lang, split) if task == "sib" else belebele_part(lang, split)
    exs = build_examples(task, df)
    scores = score_options(model, tokenizer, exs)
    pred = scores.argmax(1)
    gold = np.array([e["gold"] for e in exs])
    m = {"acc": accuracy_score(gold, pred) * 100, "f1": f1_score(gold, pred, average="macro") * 100, "n": len(exs)}
    opts = exs[0]["options"]
    out = pd.DataFrame({"id": [e["id"] for e in exs], "gold": [opts[g] for g in gold], "pred": [opts[p] for p in pred]})
    for j, o in enumerate(opts):
        out[f"logp_{o}"] = scores[:, j].round(4)
    if with_generation:
        gens = generate_answers(model, tokenizer, exs)
        parsed = [parse_generation(g, opts) for g in gens]
        m["gen_valid"] = float(np.mean([exact for _, exact in parsed]) * 100)  # exact answer format
        m["gen_acc"] = float(np.mean([p == g for (p, _), g in zip(parsed, gold)]) * 100)  # lenient parse
        out["generation"] = gens
    if run_id:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(PRED_DIR / f"{run_id}__{task}_{lang}_{split}.csv", index=False)
    return m


# ---------------------------------------------------------------------------------------------
# Model loading + LoRA
# ---------------------------------------------------------------------------------------------
def load_model(name: str, load_in_4bit: bool = True, max_seq_length: int = 2048):
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, dtype=None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


TARGETS = {
    "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def add_lora(model, r=16, lora_alpha=16, target="all", seed=3407):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model, r=r, lora_alpha=lora_alpha, lora_dropout=0, bias="none", target_modules=TARGETS[target],
        use_gradient_checkpointing="unsloth", random_state=seed, use_rslora=False, loftq_config=None)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # bitsandbytes 4-bit stores 2 weights per byte -> numel() is half the real count
    total = sum(p.numel() * (2 if getattr(p, "quant_type", None) else 1) for p in model.parameters())
    return trainable, total


def tokenize_for_training(tokenizer, examples, max_len):
    """input_ids = prompt + completion; labels = -100 on the prompt (loss on the answer only,
    same idea as Unsloth's train_on_responses_only)."""
    feats = []
    for e in examples:
        p, c = render_completion(tokenizer, e["user"], e["answer"])
        p_ids, c_ids = _encode(tokenizer, p), _encode(tokenizer, c)
        ids = (p_ids + c_ids)[-max_len:]
        labels = ([-100] * len(p_ids) + c_ids)[-max_len:]
        feats.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return feats


# ---------------------------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------------------------
def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def versions():
    v = {"python": platform.python_version()}
    for pkg in ["torch", "transformers", "peft", "unsloth", "bitsandbytes", "trl"]:
        try:
            v[pkg] = __import__(pkg).__version__
        except Exception:
            pass
    return json.dumps(v)


def log_run(row: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {**row, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if RUNS_CSV.exists():
        df = pd.read_csv(RUNS_CSV)
        df = df[df["run_id"] != row["run_id"]]  # re-running a run_id replaces its row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    cols = [c for c in RUN_COLUMNS if c in df.columns] + [c for c in df.columns if c not in RUN_COLUMNS]
    df[cols].to_csv(RUNS_CSV, index=False)


def load_runs() -> pd.DataFrame:
    return pd.read_csv(RUNS_CSV) if RUNS_CSV.exists() else pd.DataFrame(columns=RUN_COLUMNS)


def done(run_id: str) -> bool:
    df = load_runs()
    return bool(((df["run_id"] == run_id) & (df["status"] == "ok")).any())


def _cfg_row(cfg):
    row = {k: cfg.get(k) for k in CFG_KEYS}
    row["target_modules"] = cfg.get("target")
    row["gpu"] = gpu_name()
    return row


# ---------------------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------------------
DEFAULT_EVALS = [("sib", "hun_Latn", "dev"), ("sib", "hun_Latn", "test")]


def run_experiment(cfg: dict) -> dict:
    """cfg keys: run_id, group, kind ('eval' | 'lora'), model, load_in_4bit,
    evals (list of [task, lang, split]), generation (list of eval keys to also generate on),
    and for kind='lora': train_data ('sib_hun'), seed, r, lora_alpha, target, lr, epochs, batch_size,
    grad_accum, max_seq_length."""
    import torch
    from transformers import set_seed
    from unsloth import FastLanguageModel

    row = {**_cfg_row(cfg), "backend": "unsloth", "versions": versions()}
    set_seed(cfg.get("seed", 0) or 0)
    torch.cuda.reset_peak_memory_stats()

    # max_seq_length only truncates training examples; the model itself is loaded with room for the
    # longest evaluation prompt (Belebele passages are up to ~600 tokens).
    model, tok = load_model(cfg["model"], cfg.get("load_in_4bit", True), max(2048, cfg.get("max_seq_length", 2048)))
    row["peak_mem_load_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)

    if cfg["kind"] == "lora":
        from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
        from datasets import Dataset
        model = add_lora(model, cfg["r"], cfg["lora_alpha"], cfg["target"], seed=cfg["seed"])
        tr, tot = count_params(model)
        row.update(trainable_params=tr, total_params=tot, trainable_pct=round(100 * tr / tot, 4))
        task, lang = cfg["train_data"].split("_")  # e.g. 'sib_hun'
        exs = build_examples(task, load_sib(LANGS[lang], "train"))
        ds = Dataset.from_list(tokenize_for_training(tok, exs, cfg.get("max_seq_length", 2048)))
        total_steps = math.ceil(len(ds) / (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
        args = TrainingArguments(
            output_dir=str(HERE / "adapters" / cfg["run_id"]), per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"],
            learning_rate=cfg["lr"], warmup_steps=max(1, round(0.05 * total_steps)),
            lr_scheduler_type="linear", weight_decay=0.01,
            optim="adamw_8bit", logging_steps=5, save_strategy="no", report_to="none", seed=cfg["seed"],
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            dataloader_num_workers=0, remove_unused_columns=False)
        trainer = Trainer(model=model, args=args, train_dataset=ds,
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = trainer.train()
        torch.cuda.synchronize()
        row["train_time_s"] = round(time.time() - t0, 1)
        row["train_steps"] = res.global_step
        row["train_loss"] = round(res.training_loss, 4)
        row["peak_mem_train_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        row["peak_mem_alloc_train_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trainer.state.log_history).to_csv(LOG_DIR / f"{cfg['run_id']}_trainlog.csv", index=False)
    FastLanguageModel.for_inference(model)

    t0 = time.time()
    gen_keys = set(cfg.get("generation", []))
    evals = [tuple(e) for e in cfg.get("evals", DEFAULT_EVALS)]
    # Belebele 'heldout' is a subset of 'full': if both are requested, score 'full' once and
    # take the held-out passages from its predictions (same model, same prompts -> same numbers).
    evals.sort(key=lambda e: e[2] == "heldout")
    for task, lang, split in evals:
        key = f"{task}_{lang.split('_')[0]}_{split}"
        full_pred = PRED_DIR / f"{cfg['run_id']}__{task}_{lang}_full.csv"
        if task == "belebele" and split == "heldout" and (task, lang, "full") in evals and full_pred.exists():
            p = pd.read_csv(full_pred)
            p = p[p["id"].str.split("#").str[0].isin(set(belebele_split()["heldout_links"]))]
            p.to_csv(PRED_DIR / f"{cfg['run_id']}__{task}_{lang}_{split}.csv", index=False)
            m = {"acc": (p.gold == p.pred).mean() * 100, "f1": float("nan")}
        else:
            m = evaluate(model, tok, task, lang, split, run_id=cfg["run_id"], with_generation=key in gen_keys)
        row[f"{key}_acc"] = round(m["acc"], 2)
        if task == "sib" and split == "test":
            row[f"{key}_f1"] = round(m["f1"], 2)
        if "gen_acc" in m:
            row[f"{key}_gen_acc"] = round(m["gen_acc"], 2)
            row[f"{key}_gen_valid"] = round(m["gen_valid"], 2)
        print(f"  {key}: acc={m['acc']:.2f} f1={m['f1']:.2f}"
              + (f" gen_acc={m['gen_acc']:.1f} valid={m['gen_valid']:.1f}" if "gen_acc" in m else ""), flush=True)
    row["eval_time_s"] = round(time.time() - t0, 1)
    row["status"] = "ok"
    return row


def launch(cfg: dict, force: bool = False) -> dict | None:
    """Run one experiment in a fresh subprocess. If run_id is already in runs.csv, print its stored results."""
    if not force and done(cfg["run_id"]):
        row = load_runs().set_index("run_id").loc[cfg["run_id"]]
        accs = ", ".join(f"{k[:-4]}={row[k]:.2f}" for k in row.index if k.endswith("_acc") and pd.notna(row[k]))
        print(f"[already run, {row['timestamp']}] {cfg['run_id']}: {accs}")
        return row.to_dict()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cfg['run_id']}.log"
    print(f"[run] {cfg['run_id']} ...", flush=True)
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"}
    with open(log_path, "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, str(HERE / "m2_helper.py"), "--config", json.dumps(cfg)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in tail:
        if line.startswith("  ") and ": acc=" in line:
            print(line)
    if r.returncode != 0 and not done(cfg["run_id"]):
        df = load_runs()
        if not ((df["run_id"] == cfg["run_id"]) & (df["status"] == "failed")).any():
            log_run({**_cfg_row(cfg), "status": "failed", "error": f"exit code {r.returncode}"})
        print(f"[failed] {cfg['run_id']} (see {log_path.name}): " + " | ".join(tail[-3:]))
    else:
        print(f"[done] {cfg['run_id']} in {(time.time() - t0) / 60:.1f} min")
    df = load_runs()
    hit = df[df["run_id"] == cfg["run_id"]]
    return hit.iloc[-1].to_dict() if len(hit) else None


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.loads(ap.parse_args().config)
    try:
        log_run(run_experiment(cfg))
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        oom = "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__
        log_run({**_cfg_row(cfg), "status": "failed",
                 "error": ("CUDA out of memory: " if oom else "") + f"{type(e).__name__}: {str(e)[:300]}"})
        sys.exit(1)


if __name__ == "__main__":
    _main()
