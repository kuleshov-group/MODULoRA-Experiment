import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--adapter', type=str, help='adapter ID for huggingface', required=True)
parser.add_argument('--file_name', type=str, help='backup file name', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter Name: ', args.adapter)
print('Output file:', args.file_name)
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


os.environ["WANDB_DISABLED"] = "true"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False,,model_max_length=250))
tokenizer = fix_tokenizer(tokenizer)

dataset = load_dataset('samsum')
val_records = dataset['test']

## Config for llama model
model_type = "causal"
templates_path = "llama_lora_samsum.json"
only_target_loss = False
mode = "instruct"


## Load pretrained Model
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map=device_map,
    )
    
# Set model generation params
model = fix_model(model, tokenizer, use_resize=False)
model.config.num_beams = 5



ft_model_name = args.adapter

peft_model_id = f"{ft_model_name}"


model = PeftModel.from_pretrained(model, peft_model_id)

# Metric
metric = evaluate.load("rouge")



def evaluate_peft_model(sample,max_target_length=45):
    # Load dataset from the hub and get a sample
    sample_word = f"### Summarize this: {sample}\n ### Output: "
    with torch.inference_mode(), torch.autocast("cuda"):
        input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
        outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 45)
        output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
        print(f"Output:\n{output}")
    # Some simple post-processing
    return output


# run predictions
# this can take ~3-24 hours
predictions = []
for sample in tqdm(dataset['test']['dialogue']):
    p = evaluate_peft_model(sample)
    predictions.append(p)

# compute metric
rogue = metric.compute(predictions=predictions, references=dataset['test']['summary'], use_stemmer=True)

# print results
print(f'Seed: {seed}')
print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
print(f"rouge2: {rogue['rouge2']* 100:2f}%")
print(f"rougeL: {rogue['rougeL']* 100:2f}%")
print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

file_name = args.file_name
with open(file_name, 'w') as f:
    f.write(f'Seed: {seed}')
    f.write(f"Rogue1: {rogue['rouge1']* 100:2f}%")
    f.write(f"rouge2: {rogue['rouge2']* 100:2f}%")
    f.write(f"rougeL: {rogue['rougeL']* 100:2f}%")
    f.write(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")
