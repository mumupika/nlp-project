import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets

# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    # TODO: add your model arguments here
    pass


@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    # TODO: add your data arguments here
    
# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments,TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=sys.argv)
    
    # TODO Step 2: Load tokenizer and model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,device_map = 'auto')

    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    dataset = datasets.load_dataset(data_args.dataset_path)
    raw_train_dataset = dataset['train']

    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the data (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    #
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        """
        inputs = tokenizer([f"instuction: {item['instruction']},input:{'' if item['input']==None else item['input']}" for item in batch],
                           truncation=True, max_length=432, padding_side='left',padding='max_length',return_tensors='pt').to(device)
        outputs = tokenizer([f"output:{item['output']}" for item in batch],
                           truncation=True, max_length=432, padding_side='left',padding='max_length',return_tensors='pt').to(device)
        
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = outputs['input_ids']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        

    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=raw_train_dataset
    )

    # Step 6: Train!
    trainer.train()

if __name__=='__main__':
    # sys.argv = [
    # "--model_name_or_path", "./model/input/qwen2.5/transformers/0.5b/1",
    # "--torch_dtype", "bfloat16",
    # "--output_dir","./output",
    # "--dataset_path","./Dataset/alpaca-language-instruction-training",
    # "--remove_unused_columns", "False",
    # "--per_device_train_batch_size", "8",
    # "--do_train", "True",
    # "--dataloader_pin_memory","False",
    # "--logging_dir","./log"
    # ]
    # finetune()
    ### test for inference.

    device = "cpu"
    base_dir = './model/input/qwen2.5/transformers/0.5b/1'
    checkpoint_dir ='./output/checkpoint-4500'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    input_seqs = ["You are now given a task with instructions and inputs. Instructions: give me 3 advice to stay health. Input:"]
    inputs = tokenizer(input_seqs, padding='max_length',padding_side='left',return_tensors="pt",max_length = 784)
    generated_ids = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
        top_p=0.95,
        temperature=0.9,
        do_sample=True
    )

    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(generated_text)
    