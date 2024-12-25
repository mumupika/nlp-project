# Supervised Finetune Training

This is the Introduction of our repository, for the project-1 of natural language process (CS3602).

## File Structure

```c
.
├── Dataset
│   └── .gitignore // gitignored. Using alpaca-cleaned.
├── README.md
├── assets
│   ├── Comparison.png
│   └── Dataset_results.xlsx
├── auto_clean.py   // cleaning checkpoints during training.
├── eval.py         // To use for some case-study.
├── evaluation_results
│   ├── base_eval   // base model evaluation results.
│   │   └── 20241223_180428
│   │       └── summary
│   │           ├── summary_20241223_180428.csv
│   │           ├── summary_20241223_180428.md
│   │           └── summary_20241223_180428.txt
│   ├── evals_masked_sft    // output loss only results.
│   │   └── 20241223_174211
│   │       └── summary
│   │           ├── summary_20241223_174211.csv
│   │           ├── summary_20241223_174211.md
│   │           └── summary_20241223_174211.txt
│   └── evals_unmasked_sft  // whole sequence loss results.
│       └── 20241224_171102
│           └── summary
│               ├── summary_20241224_171102.csv
│               ├── summary_20241224_171102.md
│               └── summary_20241224_171102.txt
├── finetune_masked.ipynb   // calculate output loss only.
├── finetune_unmasked.ipynb // calculate the whole sequence.
├── model
│   └── .gitignore // Will be given by outer link.
├── report.pdf
└── tensorboard_events
    ├── masked_finetune_tfb // output loss only result.
    │   └── events.out.tfevents.1734937325.autodl-container-eac843a08e-1b614e77.19787.15
    └── unmasked_finetune_tfb   // whole sequence loss result.
        └── events.out.tfevents.1735021534.autodl-container-86b04fa436-52cbd070.3065.0

```

## report

Presented in `report.pdf`.

## Finetuned model link

We will provide links for our **output-loss only finetuned** model and **whole sequence model** by the following links(named **masked** and **unmasked**):

[download links by Baidu NetDisk, password 1234](https://pan.baidu.com/s/1o4LLaOw-bQMsreTjEfXlOQ?pwd=1234)

[download links by SJTU jBox, students and staff only, no dataset and base model](https://jbox.sjtu.edu.cn/l/q1hwDo)

## Important notifications

1. The finetune-evaluation code is `finetune_masked.ipynb` and `finetuned_unmasked.ipynb`, for the previous one is for output loss only finetune and the second one is for whole sequence loss calculation.
2. Logs during training process was in `tensorboard_events`.
3. The final evaluation results was in `evaluation_results`. The final statistical tables and pictures was in `assets`.
4. After you downloaded the model, you can put the model at the following structure:

```c
.
├── input       // The base model.
│   └── qwen2.5
│       └── transformers
│           └── 0.5b
│               └── 1
│                   ├── LICENSE
│                   ├── README.md
│                   ├── config.json
│                   ├── generation_config.json
│                   ├── merges.txt
│                   ├── model.safetensors
│                   ├── tokenizer.json
│                   ├── tokenizer_config.json
│                   └── vocab.json
├── masked      // The output-only model.
│   └── checkpoint-38820
│       ├── added_tokens.json
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── optimizer.pt
│       ├── rng_state.pth
│       ├── scheduler.pt
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── trainer_state.json
│       ├── training_args.bin
│       └── vocab.json
└── unmasked        // The whole seq loss.
    └── checkpoint-38820
        ├── added_tokens.json
        ├── config.json
        ├── generation_config.json
        ├── merges.txt
        ├── model.safetensors
        ├── optimizer.pt
        ├── rng_state.pth
        ├── scheduler.pt
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── trainer_state.json
        ├── training_args.bin
        └── vocab.json
```
