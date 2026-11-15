"""Milestone 1: reproduce the SIB-200 XLM-R base results for any language.

This is the same pipeline as SIB200_Milestone1_Final.ipynb, as a standalone script that
works outside Colab (local paths, any language):

  1. clone the original SIB-200 repository (into ../external/sib-200) if it is missing
  2. apply the four compatibility fixes + the BF16 autocast patch
  3. train XLM-R base with the authors' glot.sh settings for seeds 1-5
     (--do_train --do_eval --do_predict, exactly like the authors' scripts)
  4. read the authors' per-seed results for <lang>_xlmrbase from outputs.zip and compare

Usage:
    python run_xlmr_sib200.py --lang hun_Latn
    python run_xlmr_sib200.py --lang hun_Latn --train_batch 8 --grad_accum 4   # same effective batch 32, less memory

Outputs (milestone1/results/):
    xlmr_<lang>_ours.csv     per-seed test accuracy / F1 of our runs
    xlmr_<lang>_authors.csv  per-seed test accuracy / F1 from outputs.zip
    xlmr_<lang>_summary.json mean +- std for both, plus run settings
"""
import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SIB_DIR = REPO_ROOT / "external" / "sib-200"
RESULTS_DIR = HERE / "results"
CKPT_DIR = HERE / "checkpoints"


def ensure_sib200():
    if not SIB_DIR.exists():
        SIB_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/dadelani/sib-200.git", str(SIB_DIR)], check=True)
    return subprocess.check_output(["git", "-C", str(SIB_DIR), "rev-parse", "HEAD"], text=True).strip()


def apply_fixes():
    """The four required fixes + BF16 autocast, identical to the Milestone 1 notebook."""
    code_dir = SIB_DIR / "code"
    subprocess.run(["git", "-C", str(SIB_DIR), "restore", "code/train_textclass.py", "code/util_textclass.py"], check=True)

    train_file = code_dir / "train_textclass.py"
    util_file = code_dir / "util_textclass.py"
    train = train_file.read_text(encoding="utf-8")
    util = util_file.read_text(encoding="utf-8")

    # Fix 1: AdamW from PyTorch
    train, n = re.subn(r"(?m)^[ \t]*AdamW,[ \t]*\n", "", train)
    assert n == 1, "AdamW import not found"
    train = "from torch.optim import AdamW\n" + train
    # Fix 2: encode_plus -> direct tokenizer call
    assert util.count("tokenizer.encode_plus(text=instance.text,") == 1
    util = util.replace("tokenizer.encode_plus(text=instance.text,", "tokenizer(text=instance.text,")
    # Fix 3: pad_to_max_length -> padding
    assert util.count("pad_to_max_length=True") == 1
    util = util.replace("pad_to_max_length=True", 'padding="max_length"')
    # Fix 4: torch.load of our own cached features
    assert train.count("torch.load(cached_features_file)") == 1
    train = train.replace("torch.load(cached_features_file)", "torch.load(cached_features_file, weights_only=False)")

    train_file.write_text(train, encoding="utf-8")
    util_file.write_text(util, encoding="utf-8")

    # BF16 autocast copy (forward pass + loss only; optimizer, clipping unchanged)
    amp, n = re.subn(
        r"^([ \t]*)outputs = model\(\*\*inputs\)\n[ \t]*loss = outputs\[0\]",
        lambda m: (f"{m.group(1)}with torch.autocast('cuda', dtype=torch.bfloat16):\n"
                   f"{m.group(1)}    outputs = model(**inputs)\n"
                   f"{m.group(1)}    loss = outputs[0]"),
        train, count=1, flags=re.MULTILINE)
    assert n == 1, "BF16 patch failed"
    amp_file = code_dir / "train_textclass_amp.py"
    amp_file.write_text(amp, encoding="utf-8")
    return amp_file


def read_metrics(path):
    m = {}
    for line in Path(path).read_text().splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            m[k.strip()] = float(v)
    return m


def authors_results(lang):
    rows = []
    with zipfile.ZipFile(SIB_DIR / "outputs.zip") as z:
        for seed in range(1, 6):
            name = f"outputs/{lang}_xlmrbase/test_result_{lang}_{seed}.txt"
            text = z.read(name).decode()
            m = {k.strip(): float(v) for k, v in (l.split(" = ") for l in text.splitlines() if " = " in l)}
            rows.append({"seed": seed, "test_acc": m["acc"] * 100, "test_f1": m["f1"] * 100})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="hun_Latn")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--seeds", default="1,2,3,4,5")
    # Authors' glot.sh settings
    p.add_argument("--max_seq_length", type=int, default=164)
    p.add_argument("--train_batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--keep_checkpoints", action="store_true")
    args = p.parse_args()

    import torch
    assert torch.cuda.is_available(), "A CUDA GPU is required"
    gpu = torch.cuda.get_device_name(0)

    commit = ensure_sib200()
    script = apply_fixes()
    data_dir = SIB_DIR / "data" / "annotated" / args.lang
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        out_dir = CKPT_DIR / f"{args.lang}_xlmrbase_seed{seed}"
        cmd = [sys.executable, str(script),
               "--data_dir", str(data_dir), "--model_type", "xlmroberta",
               "--model_name_or_path", args.model, "--output_dir", str(out_dir),
               "--output_result", "test_result", "--output_prediction_file", "test_predictions",
               "--labels", str(data_dir / "labels.txt"),
               "--max_seq_length", str(args.max_seq_length),
               "--num_train_epochs", str(args.epochs), "--learning_rate", str(args.lr),
               "--per_gpu_train_batch_size", str(args.train_batch),
               "--per_gpu_eval_batch_size", str(args.train_batch),
               "--gradient_accumulation_steps", str(args.grad_accum),
               "--save_steps", "500000", "--seed", str(seed),
               "--do_train", "--do_eval", "--do_predict", "--overwrite_output_dir"]
        print(f"[{args.lang}] seed {seed} ...", flush=True)
        t0 = time.time()
        with open(log_dir / f"{args.lang}_seed{seed}.log", "w", encoding="utf-8") as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=SIB_DIR / "code")
        minutes = (time.time() - t0) / 60
        if r.returncode != 0:
            print((log_dir / f"{args.lang}_seed{seed}.log").read_text(encoding="utf-8")[-3000:])
            raise RuntimeError(f"seed {seed} failed")
        m = read_metrics(out_dir / "test_result.txt")
        rows.append({"seed": seed, "test_acc": m["acc"] * 100, "test_f1": m["f1"] * 100, "minutes": round(minutes, 2)})
        shutil.copy(out_dir / "test_predictions.txt", RESULTS_DIR / f"xlmr_{args.lang}_test_predictions_seed{seed}.txt")
        print(f"  acc={rows[-1]['test_acc']:.2f}  ({minutes:.1f} min)", flush=True)
        if not args.keep_checkpoints:
            shutil.rmtree(out_dir, ignore_errors=True)

    ours = pd.DataFrame(rows)
    theirs = authors_results(args.lang)
    ours.to_csv(RESULTS_DIR / f"xlmr_{args.lang}_ours.csv", index=False)
    theirs.to_csv(RESULTS_DIR / f"xlmr_{args.lang}_authors.csv", index=False)

    summary = {
        "lang": args.lang, "gpu": gpu, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "sib200_commit": commit,
        "settings": {k: getattr(args, k) for k in ["model", "max_seq_length", "train_batch", "grad_accum", "epochs", "lr"]},
        "effective_batch": args.train_batch * args.grad_accum,
        "ours_mean_acc": ours.test_acc.mean(), "ours_std_acc": ours.test_acc.std(),
        "authors_mean_acc": theirs.test_acc.mean(), "authors_std_acc": theirs.test_acc.std(),
        "ours_mean_f1": ours.test_f1.mean(), "authors_mean_f1": theirs.test_f1.mean(),
    }
    (RESULTS_DIR / f"xlmr_{args.lang}_summary.json").write_text(json.dumps(summary, indent=2))
    print(ours.round(2).to_string(index=False))
    print(f"Ours:    {summary['ours_mean_acc']:.2f} +- {summary['ours_std_acc']:.2f}")
    print(f"Authors: {summary['authors_mean_acc']:.2f} +- {summary['authors_std_acc']:.2f}")


if __name__ == "__main__":
    main()
