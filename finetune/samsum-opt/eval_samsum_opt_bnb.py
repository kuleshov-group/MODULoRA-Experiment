import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--adapter', type=str, help='adapter in huggingface', required=True)
parser.add_argument('--file_name', type=str, help='backup file name', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)
parser.add_argument('--bit', type=str, help='8bit or 4bit', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter Name: ', args.adapter)
print('Output file:', args.file_name)
print('Seed: ', args.seed)
print('Bit: ', args.bit)

import random
import json
import os

# import wandb
import torch
import numpy as np
# import bitsandbytes as bnb
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training, PeftModel
from datasets import load_dataset

from utils import *
from data import *

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm


output_dir = args.adapter
model_name = args.model_name
seed = args.seed
train_sample_rate = 1.0
val_sample_rate = 1.0
local_rank = 0

set_random_seed(seed)
logging.set_verbosity_info()

# with open(config_file, "r") as r:
#     config = json.load(r)

os.environ["WANDB_DISABLED"] = "true"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer = fix_tokenizer_opt(tokenizer)
# tokenizer.save_pretrained(output_dir)

dataset = load_dataset('samsum')
val_records = dataset['test']

## Config for llama 7-b
model_type = "causal"
templates_path = "llama_lora_samsum.json"
only_target_loss = False
mode = "instruct"

model_types = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM
}


if args.bit == '4bit':
    load_in_8bit = False
    load_in_4bit = True
else:
    load_in_8bit = True
    load_in_4bit = False


if load_in_8bit:
    assert not load_in_4bit
    model = model_types[model_type].from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map=device_map
    )
elif load_in_4bit:
    assert not load_in_8bit
    #use_bf16 = trainer_config.get("bf16", False)
    use_bf16 = True
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = model_types[model_type].from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4"
        ),
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32
    )
else:
    model = model_types[model_type].from_pretrained(model_name)

# Default model generation params
model = fix_model(model, tokenizer, use_resize=False)
model.config.num_beams = 5


## load in adapter
ft_model_name = args.adapter
peft_model_id = f"{ft_model_name}"
model = PeftModel.from_pretrained(model, peft_model_id)

# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model(sample,max_target_length=45):
    # Load dataset from the hub and get a sample
    sample_word = f"### Summarize this: {sample}\n ### Output: "
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.inference_mode(), torch.autocast("cuda"):
        outputs = model.generate(input_ids=input_ids, do_sample=True,max_new_tokens = 45)
    output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
    print(f"Output:\n{output}")
    # Some simple post-processing
    return output

# run predictions
# this can take ~45 minutes
predictions = []
for sample in tqdm(dataset['test']['dialogue']):
    p = evaluate_peft_model(sample)
    predictions.append(p)

# compute metric



file_name = args.file_name
# with open(file_name, 'w') as f:
#     for item in predictions:
#         # write each item on a new line
#         f.write("%s\n" % item)
#     f.write(f'Seed: {seed}')


def process_file(filename):
    output_list = []
    delete_lines = False
    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line.startswith("### Summarize this:"):
                delete_lines = True
                continue
            elif stripped_line.startswith("### Output: "):
                output = stripped_line[len("### Output: "):]
                output_list.append(output)
                delete_lines = False
                continue

            if not delete_lines:
                output_list.append(stripped_line)

    return output_list

# predictions = process_file(file_name)
# predictions.pop()

rogue = metric.compute(predictions=predictions, references=dataset['test']['summary'], use_stemmer=True)

# print results
print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

with open(file_name, 'w') as f:
    for item in predictions:
        # write each item on a new line
        f.write("%s\n" % item)
    f.write(f'Seed: {seed}\n')
    f.write(f"Rogue1: {rogue['rouge1']* 100:2f}%\n")
    f.write(f"rouge2: {rogue['rouge2']* 100:2f}%\n")
    f.write(f"rougeL: {rogue['rougeL']* 100:2f}%\n")
    f.write(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%\n")