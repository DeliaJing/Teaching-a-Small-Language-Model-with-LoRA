## Milestone 1

| language   | ours (5 seeds)   | authors, outputs.zip (5 seeds)   |
|:-----------|:-----------------|:---------------------------------|
| eng_Latn   | 89.9 ± 1.1       | 90.0 ± 0.7                       |
| hun_Latn   | 86.3 ± 1.0       | 86.0 ± 1.0                       |

## GPT reference

| model   | lang     |   acc |   no label found (%) |
|:--------|:---------|------:|---------------------:|
| gpt3.5  | hun_Latn |  62.7 |                 24.5 |
| gpt3.5  | eng_Latn |  73.5 |                 11.8 |
| gpt4    | hun_Latn |  80.9 |                  5.4 |
| gpt4    | eng_Latn |  81.4 |                  1.5 |

## Main results

| model             | method                      |   runs | SIB hun test acc   | SIB hun test F1   | SIB eng test acc   | trainable params   | Belebele hun held-out acc   | Belebele eng held-out acc   | peak train mem, allocated (GB)   | train time (min)   |
|:------------------|:----------------------------|-------:|:-------------------|:------------------|:-------------------|:-------------------|:----------------------------|:----------------------------|:---------------------------------|:-------------------|
| XLM-R base (278M) | full fine-tuning (M1)       |      5 | 86.3 ± 1.0         | 86.0 ± 1.0        | 89.9 ± 1.1         | 278M               | –                           | –                           | –                                | –                  |
| Qwen3-1.7B        | zero-shot (4-bit)           |      1 | 48.0               | 41.0              | 50.5               | –                  | 37.5                        | 56.2                        | –                                | –                  |
| Qwen3-1.7B        | zero-shot (16-bit)          |      1 | 32.8               | 21.3              | 48.0               | –                  | 44.5                        | 72.2                        | –                                | –                  |
| Qwen3-1.7B        | QLoRA, best config, SIB hun |      3 | 87.8 ± 0.0         | 86.9 ± 0.4        | 91.2 ± 1.0         | 17.4M              | 52.2 ± 2.7                  | 76.4 ± 1.8                  | 3.0 ± 0.0                        | 2.2 ± 0.0          |

## Sweep

| run_id                | status   | notes               |   trainable (M) |   sib_hun_dev_acc |   sib_hun_test_acc |   sib_hun_test_f1 |   peak_mem_alloc_train_gb |   train_time_s |   train_loss |   error |
|:----------------------|:---------|:--------------------|----------------:|------------------:|-------------------:|------------------:|--------------------------:|---------------:|-------------:|--------:|
| qwen3-1.7b_sw_default | ok       | default             |           17.43 |             78.79 |              86.76 |             86.24 |                     2.956 |          138.6 |       0.3151 |     nan |
| qwen3-1.7b_sw_r4      | ok       | r=4, lora_alpha=4   |            4.36 |             77.78 |              84.31 |             83.3  |                     2.834 |          201.4 |       0.4014 |     nan |
| qwen3-1.7b_sw_r8      | ok       | r=8, lora_alpha=8   |            8.72 |             81.82 |              85.29 |             84.57 |                     2.875 |          154.3 |       0.3698 |     nan |
| qwen3-1.7b_sw_r32     | ok       | r=32, lora_alpha=32 |           34.87 |             79.8  |              87.25 |             86.56 |                     3.119 |          149.8 |       0.282  |     nan |
| qwen3-1.7b_sw_r64     | ok       | r=64, lora_alpha=64 |           69.73 |             83.84 |              87.25 |             86.39 |                     3.473 |          153.9 |       0.2999 |     nan |
| qwen3-1.7b_sw_r16_a32 | ok       | lora_alpha=32       |           17.43 |             79.8  |              85.78 |             84.75 |                     2.956 |          143.1 |       0.293  |     nan |
| qwen3-1.7b_sw_attn    | ok       | target=attn         |            6.42 |             82.83 |              84.31 |             84.2  |                     2.853 |          148.5 |       0.3814 |     nan |
| qwen3-1.7b_sw_16bit   | ok       | load_in_4bit=False  |           17.43 |             82.83 |              87.75 |             86.99 |                     4.852 |          124.4 |       0.2399 |     nan |
| qwen3-1.7b_sw_lr1e-4  | ok       | lr=0.0001           |           17.43 |             81.82 |              84.31 |             83.4  |                     2.956 |          131.6 |       0.377  |     nan |
| qwen3-1.7b_sw_lr5e-4  | ok       | lr=0.0005           |           17.43 |             85.86 |              87.75 |             86.91 |                     2.956 |          131.8 |       0.2885 |     nan |
| qwen3-1.7b_sw_ep1     | ok       | epochs=1            |           17.43 |             82.83 |              81.37 |             78.81 |                     2.956 |           72.5 |       0.5228 |     nan |
| qwen3-1.7b_sw_ep3     | ok       | epochs=3            |           17.43 |             81.82 |              88.24 |             87.44 |                     2.956 |          191.5 |       0.2267 |     nan |

## Format following

| run_id                      |   sib_hun_test_acc |   sib_hun_test_gen_acc |   sib_hun_test_gen_valid |   belebele_hun_full_acc |   belebele_hun_full_gen_acc |   belebele_hun_full_gen_valid |
|:----------------------------|-------------------:|-----------------------:|-------------------------:|------------------------:|----------------------------:|------------------------------:|
| base_qwen3-1.7b_4bit_zs_en  |              48.04 |                  37.75 |                     0    |                   38.89 |                       50.11 |                             0 |
| base_qwen3-1.7b_16bit_zs_en |              32.84 |                  47.06 |                    39.71 |                   47.78 |                       46.89 |                             0 |

