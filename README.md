# Supervised Finetune Training

This is the Introduction of our repository, for the project-1 of natural language process (CS3602).

## Result

### Pure sft full finetune

![img](./nlp-project1/assets/Comparison.png)

### Lora fine tune 1.5B

![img](./nlp-project1/assets/classical_lora.png)

## To developers

For those who maintaining the project, please be aware that you should work under the directories that you are **respond to**.

## File Structure

### root

```bash
.
├── Dataset     # 数据集存放
├── README.md   # README
├── model   # 模型存放
├── nlp-project1    # 项目一
├── nlp-project2    # 项目二
└── report.pdf  # 报告
```

### nlp-project1

```bash
.
├── assets  # 实验的统计数据表格和结果图片
│   ├── Comparison.png
│   ├── Dataset_results.xlsx
│   ├── Lora_results_on_1.5B.xlsx
│   ├── Parameter.png
│   ├── all_lora.png
│   ├── classical_lora.png
│   └── parameter.xlsx
├── auto_clean.py   # 训练过程中清楚checkpoint脚本
├── eval.py     # 简单的评估脚本
├── evaluation_results  # opencompass评测结果
│   ├── LoRA_finetuned_eval
│   │   └── 20241231_210406
│   │       └── summary
│   │           ├── summary_20241231_210406.csv
│   │           ├── summary_20241231_210406.md
│   │           └── summary_20241231_210406.txt
│   ├── base_1.5B_eval
│   │   └── 20241231_200238
│   │       └── summary
│   │           ├── summary_20241231_200238.csv
│   │           ├── summary_20241231_200238.md
│   │           └── summary_20241231_200238.txt
│   ├── base_eval
│   │   └── 20241223_180428
│   │       └── summary
│   │           ├── summary_20241223_180428.csv
│   │           ├── summary_20241223_180428.md
│   │           └── summary_20241223_180428.txt
│   ├── evals_masked_sft
│   │   └── 20241223_174211
│   │       └── summary
│   │           ├── summary_20241223_174211.csv
│   │           ├── summary_20241223_174211.md
│   │           └── summary_20241223_174211.txt
│   └── evals_unmasked_sft
│       └── 20241224_171102
│           └── summary
│               ├── summary_20241224_171102.csv
│               ├── summary_20241224_171102.md
│               └── summary_20241224_171102.txt
├── finetune_masked.ipynb   # 全序列输出SFT微调
├── finetune_unmasked.ipynb # output-onlySFT微调
└── peft.ipynb  # peft lora微调
```

### nlp-project2

```bash
.
├── README.md #相关的使用方法和注意事项
├── main.py #聊天机器人主程序
├── model #聊天机器人实现
│   ├── __init__.py
│   ├── api_model.py
│   ├── base.py
│   ├── character_settings.yaml
│   └── local_model.py
├── model_test #聊天机器人测试程序和结果
│   ├── ans_character.txt
│   ├── ans_db.txt
│   ├── ans_harm.txt
│   ├── ans_info.txt
│   ├── ans_know.txt
│   ├── test_character.py
│   ├── test_db.py
│   └── test_model.py
├── requirements.txt #环境配置文件
└── server #一个简单的聊天机器人服务器
    ├── server.py
    └── web
        ├── Chat.js
        ├── WebClientPage.html
        └── style.css
```

## Finetuned model link

For Qwen2.5-0.5B:

We will provide links for our **output-loss only finetuned** model and **whole sequence model** by the following links(named **masked** and **unmasked**):

[download links by Baidu NetDisk, password 1234](https://pan.baidu.com/s/1o4LLaOw-bQMsreTjEfXlOQ?pwd=1234)

[download links by SJTU jBox, students and staff only, no dataset and base model](https://jbox.sjtu.edu.cn/l/q1hwDo)

For Qwen2.5-1.5B:

We will provide links for our **lora finetuned** model by the following links:

[download links by SJTU jBox, students and staff only, no base model](https://jbox.sjtu.edu.cn/l/G1lhIi)

## Important notifications

1. The finetune-evaluation code is `finetune_masked.ipynb` and `finetuned_unmasked.ipynb`, for the previous one is for output loss only finetune and the second one is for whole sequence loss calculation.
2. The final evaluation results was in `evaluation_results`. The final statistical tables and pictures was in `assets`.
3. After you downloaded the model, you can put the model at the following structure:
