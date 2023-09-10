import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='HF model name with your user', required=True)
parser.add_argument('--adapter', type=str, help='Path to store adapter weight', required=True)
parser.add_argument('--file_name', type=str, help='backup file name', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter Path: ', args.adapter)
print('Seed: ', args.seed)

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
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from datasets import load_dataset

from utils import *
from data import *

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.engine.lora.config import FinetuneConfig
from llmtune.engine.lora.peft import quant_peft
from llmtune.utils import to_half_precision

output_dir = args.adapter
seed = args.seed
train_sample_rate = 1.0
val_sample_rate = 1.0
local_rank = 0

# model config
model_name = args.model_name
tokenizer_name = "facebook/opt-6.7b"
DEV = 'cuda'

set_random_seed(42)
logging.set_verbosity_info()

# with open(config_file, "r") as r:
#     config = json.load(r)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

transformers.logging.set_verbosity_info()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token_id = 0
## Fix Tokenizer
tokenizer = fix_tokenizer_opt(tokenizer)

# load model
llm = AutoLLMForCausalLM.from_pretrained(model_name)
## Fix Model
lllm = fix_model(llm, tokenizer, use_resize=False)
llm.eval()
llm = llm.to(DEV)
llm = to_half_precision(llm)



## dataset
dataset = load_dataset('samsum')
train_records = dataset['train']
val_records = dataset['test']
#random.shuffle(train_records)
print("train_record[0]: ",train_records[0])

## Config for llama 7-b
model_type = "causal"
templates_path = "llama_lora_samsum.json"
only_target_loss = False
mode = "instruct"


adapter_path = args.adapter
model = quant_peft.PeftModel.from_pretrained(
    llm, adapter_path, 
    device_map='auto'
)
print(adapter_path, 'loaded')


# Model configs
model.config.num_beams = 5


# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model(sample,max_target_length=45):
    # Load dataset from the hub and get a sample
    sample_word = f"### Summarize this: {sample}\n ### Output: "
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode(), torch.autocast("cuda"):
    print("input_ids: ",input_ids)
    outputs = model.generate(input_ids=input_ids, do_sample=True, max_new_tokens = 45)
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


# def process_file(filename):
#     output_list = []
#     delete_lines = False
#     with open(filename, 'r') as file:
#         for line in file:
#             stripped_line = line.strip()
#             if stripped_line.startswith("### Summarize this:"):
#                 delete_lines = True
#                 continue
#             elif stripped_line.startswith("### Output: "):
#                 output = stripped_line[len("### Output: "):]
#                 output_list.append(output)
#                 delete_lines = False
#                 continue

#             if not delete_lines:
#                 output_list.append(stripped_line)

#     return output_list

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

