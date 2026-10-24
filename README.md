# Teaching a Small Language Model Your Language with LoRA

**Deep Learning Group Project — Milestone 1**

## Group Members

| Name | Neptun Code |
|---|---|
| Zhang JinJing | M5UG9O |
| Márton Németh | FXHRHX |
| Hussein Mnati Madhi | GN0GC8 |

## 1. Project Overview

In Milestone 1, we reproduce the English SIB-200 topic classification results using XLM-R base.

We use 701 training, 99 validation, and 204 test examples. Our goal is to train the model five times using different random seeds and compare the average accuracy with the authors' reported result of 90.0%.

## 2. Training Settings

First, we check the authors' original training scripts. Since the assignment mentions the Glot500 script as an example, we use the settings from `glot.sh` as our reference.

We train XLM-R base for 20 epochs with batch size 16, gradient accumulation 2 (effective batch size 32), maximum sequence length 164, and learning rate 1e-5.

We use five random seeds (1–5) and BF16 mixed precision on an NVIDIA A100 GPU to make training faster.

The exact XLM-R base training command is not available in the authors' repository, so we follow the documented settings from `glot.sh`.

## 3. Compatibility Fixes

We make four changes so the original SIB-200 code works with newer libraries:

1. Import `AdamW` from PyTorch instead of Transformers.
2. Replace `tokenizer.encode_plus()` with a direct tokenizer call.
3. Replace `pad_to_max_length=True` with `padding="max_length"`.
4. Use `torch.load(..., weights_only=False)` for trusted cached features.

## 4. Results

We train five models using the same settings and evaluate them on the English test set.

| Seed | Test Accuracy |
|---|---:|
| 1 | 89.71% |
| 2 | 88.24% |
| 3 | 89.71% |
| 4 | 90.69% |
| 5 | 91.18% |
| **Average** | **89.90%** |

The authors reported **90.00%** for English XLM-R base. Our average result is **89.90%**, only **0.10 percentage points lower**.

This shows that we reproduced a result very close to the authors' reported baseline.

## 5. How to Run

The full experiment is in `SIB200_Milestone1_Final.ipynb`.

1. Open the notebook in Google Colab.
2. Select an A100 GPU or another BF16-compatible CUDA GPU.
3. Click **Runtime → Run all**.

The notebook automatically installs the libraries, downloads the SIB-200 repository, fixes the code, trains all five models, and displays the results.

We tested the full workflow successfully on Google Colab. Results may vary slightly across environments.

## References

- [Original SIB-200 repository](https://github.com/dadelani/sib-200)
- [SIB-200 paper](https://arxiv.org/abs/2309.07445)
