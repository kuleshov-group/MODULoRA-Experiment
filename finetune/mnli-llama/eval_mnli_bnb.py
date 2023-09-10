import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--adapter_name', type=str, help='Name of the adapter on HF', required=True)
parser.add_argument('--file_name', type=str, help='backup file name', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter name:', args.adapter_name)
print('File Name:', args.file_name)
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
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training, PeftModel, PeftConfig
from datasets import load_dataset
import accelerate

from utils import *
from data_mnli_label import *

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# output_dir = args.adapter
model_name = args.model_name
seed = args.seed
train_sample_rate = 1.0
val_sample_rate = 1.0
local_rank = 0

set_random_seed(seed)
logging.set_verbosity_info()

# with open(config_file, "r") as r:
#     config = json.load(r)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=False)
tokenizer = fix_tokenizer(tokenizer)
# tokenizer.save_pretrained(output_dir)

dataset = load_dataset('multi_nli')
train_records = dataset['train']
val_records = dataset['validation_matched']
#random.shuffle(train_records)
print("train_record[0]: ",train_records[0])

## Config for llama 7-b
model_type = "causal"
templates_path = "llama_lora_mnli_label.json"
only_target_loss = False
mode = "instruct"


# model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         load_in_8bit=True,
#         device_map=device_map,
#     )
model_types = {
    "causal": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM
}
load_in_8bit = False
load_in_4bit = True
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


# from accelerate import infer_auto_device_map
# device_map={'':'0'}
#device_map = infer_auto_device_map(model, max_memory={0: "47GiB", 1: "47GiB"})


#model.load_adapter(args.weight_path, adapter_name="lora")


# from huggingface_hub import login
# access_token_read = "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"
# login(token = access_token_read)


# HUGGING_FACE_USER_NAME = "osieosie"

# ft_model_name = "Llama_lora_mnli_65B_8bit"

# peft_model_id = f"{HUGGING_FACE_USER_NAME}/{ft_model_name}"

peft_model_id = args.adapter_name
config = PeftConfig.from_pretrained(peft_model_id)

# pretrained_model_name = "huggyllama/llama-7b"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False)
# #Fix Tokenizer
# tokenizer = fix_tokenizer(tokenizer)
# #tokenizer.save_pretrained(output_dir)

# original_model = LlamaForCausalLM.from_pretrained(pretrained_model_name,
#                                                   device_map='auto',
#                                                   # offload_folder="offload",
#                                                   load_in_8bit=True)

# ##Fix Model
# original_model = fix_model(original_model, tokenizer, use_resize=False)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)
#model = PeftModel.from_pretrained(model, "osieosie/Llama_lora_mnli_65B_8bit")

print("type of model: ",type(model))
print("type of tokenizer:",type(tokenizer))



def evaluate_peft_model_mnli(sample,max_target_length=65):
    instruction, input, genre = sample['premise'], sample['hypothesis'], sample['genre']
    sample_word = f"### Premise: {instruction}\n ### Hypothesis: {input}\n ### Genre: {genre} ### Label: "
    # print(sample_word)
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.inference_mode(), torch.autocast("cuda"):
        outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 5)
    output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
    print(f"Output: {output}\n")
    # Some simple post-processing
    return output

# def evaluate_peft_model_mnli(sample,max_target_length=5):
#     instruction, input = sample['premise'], sample['hypothesis']
#     sample_word = f"### Premise: {instruction}\n ### Hypothesis: {input} \n ### Classification: "
#     # print(sample_word)
#     input_ids = tokenizer(sample_word, padding=True, truncation=True, return_tensors="pt").cuda()
#     with torch.inference_mode():
#         outputs = model.generate(**input_ids, do_sample=True, top_p=0.9, max_new_tokens = 5)
#     output = tokenizer.decode(outputs, skip_special_tokens=True).replace(sample_word,"")
#     # print(f"Output:\n{output}")
#     # Some simple post-processing
#     return output



file_name = args.file_name
# if os.path.exists(file_name):
#     with open(file_name, 'r') as predictions_file:
#         processed_samples = len(list(predictions_file))
# else:
#     processed_samples = 0

strip_word = '### Label: '
predictions = []
# Open the predictions file in append mode
# with open(file_name, 'a') as predictions_file:
#     for i in tqdm(range(processed_samples, 9815)):
#         p = evaluate_peft_model_mnli(val_records[i]).replace(strip_word,"").strip()
#         # Save the prediction to the file
#         predictions_file.write(p + "\n") 
#         predictions_file.flush()

for i in tqdm(range(9815)):
    p = evaluate_peft_model_mnli(val_records[i]).replace(strip_word,"").strip()
    predictions.append(p)

# with open(file_name, 'r') as predictions_file:
#     predictions = [line.strip() for line in predictions_file]


references_orig = val_records['label']
## convert references to list of strings
references = []
for item in references_orig:
    references.append(str(item))

acc = 0
for i in range(9815):
  if predictions[i].lower() == references[i].lower():
    acc += 1
acc /= len(predictions)

with open(file_name, 'w') as f:
    for item in predictions:
        # write each item on a new line
        f.write("%s\n" % item)
    f.write(f'Seed: {seed}')
    f.write(f"Accuracy: {acc}")
