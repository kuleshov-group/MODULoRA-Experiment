import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--weight_path', type=str, help='Path to the weights', required=True)
parser.add_argument('--adapter', type=str, help='Path to store adapter weight', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)
parser.add_argument('--file_name', type=str, help='file name to store predictions and acc', required=True)
parser.add_argument('--checkpoint_name', type=str, help='folder name to store all the check points', required=True)
parser.add_argument('--start_index', type=int, help='model seed number', required=True)
parser.add_argument('--end_index', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Weight Path:', args.weight_path)
print('Adapter Path: ', args.adapter)
print('Seed: ', args.seed)

import random
import json
import os
import pickle

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
from data_mnli_label import *

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from llmtune.executor import load_llm, load_adapter
from llmtune.engine.lora.peft import quant_peft

output_dir = args.adapter
model_name = "huggyllama/llama-13b"
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

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer = fix_tokenizer(tokenizer)
# tokenizer.save_pretrained(output_dir)

dataset = load_dataset('multi_nli')
train_records = dataset['train']
val_records = dataset['validation_matched']
#random.shuffle(train_records)
print("train_record[0]: ",train_records[0])

## Config for llama 7-b
model_type = "causal"
templates_path = "llama_lora_mnli.json"
only_target_loss = False

llmtune_model_name = args.model_name
llmtune_quantized_weights_path = args.weight_path ## probably want to change this using our version of the right way
llmtune_groupsize = 64


llm, _ = load_llm(
    llmtune_model_name,
    llmtune_quantized_weights_path,
    llmtune_groupsize
)
model = fix_model(llm, tokenizer, use_resize=False)

# Default model generation params
model.config.num_beams = 5


if not ddp and torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True


model = load_adapter(model, adapter_path=output_dir)

# Metric

def evaluate_peft_model_mnli(sample,max_target_length=65):
    instruction, input, genre = sample['premise'], sample['hypothesis'], sample['genre']
    sample_word = f"### Premise: {instruction}\n ### Hypothesis: {input}\n ### Genre: {genre} ### Label: "
    print(sample_word)
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 5)
    output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
    output = output.strip()
    print(f"Output:\n{output}")
    # Some simple post-processing
    return output



def acc_compute(predictions,references):
    acc = 0
    for i in range(len(predictions)):
        if predictions[i].lower() == references[i].lower():
            acc += 1
    acc /= len(predictions)

    print("accuracy:", acc)
    return acc


def store_pred(file_name_pickle_pred,file_name_pickle_ref,predictions,references):
    with open(file_name_pickle_pred, "wb") as fp:   #Pickling
        pickle.dump(predictions, fp)
    with open(file_name_pickle_ref, "wb") as fp:   #Pickling
        pickle.dump(references, fp)




##Arguments setting
start_index = args.start_index
end_index = args.end_index
eval_len =  end_index - start_index
eval_save_len = eval_len // 10 
print("Evaluation will start at: ", start_index)
print("Evaluation will end at: ", end_index)
print(f'Evaluation will save at every {eval_save_len} steps')


## Create Check point Folder
checkpoint_path = f'{args.checkpoint_name}_{start_index}_{end_index}'

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, checkpoint_path)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)





predictions = []
references_orig = val_records['label'][start_index:end_index]
## convert references to list of strings
references = []
for item in references_orig:
    references.append(str(item))


count_eval = 0
for idx in tqdm(range(start_index, end_index)):
    sample = val_records[idx]
    p = evaluate_peft_model_mnli(sample)
    predictions.append(p)
    count_eval += 1
    ## Detecting checkpoing
    if (count_eval%eval_save_len == 0):
       print(f'=>=>Checkpointing at {count_eval} steps<=<=')

       predictions_step = [s.strip() for s in predictions]
       print("prediction_step: ", predictions_step)
       references_step = references[0:count_eval]
       print("references_step: ", references_step)
       acc = acc_compute(predictions_step,references_step)
       checkpoint_name_txt = f'{final_directory}/{count_eval}.txt'
       checkpoint_name_pred = f'{final_directory}/{count_eval}_pred' ## pickle file for pred list
       checkpoint_name_ref = f'{final_directory}/{count_eval}_ref' ## pickle file for ref list
       ## writing pickle file
       store_pred(checkpoint_name_pred,checkpoint_name_ref,predictions_step,checkpoint_name_ref)
       with open(checkpoint_name_txt, "w") as f:
            for item in predictions_step:
                # write each item on a new line
                f.write("%s\n" % item)
            f.write("%s\n" % acc)
            


       
predictions = [s.strip() for s in predictions]



file_name = args.file_name

with open(file_name, 'w') as f:
    for item in predictions:
        # write each item on a new line
        f.write("%s\n" % item)
    f.write("%s\n" % acc)


file_name_pickle_pred = f'{final_directory}/final_pred_{start_index}_{end_index}'
file_name_pickle_ref = f'{final_directory}/final_ref_{start_index}_{end_index}'

store_pred(file_name_pickle_pred,file_name_pickle_ref,predictions,references)


"""
Loading pickle file
with open("test", "rb") as fp:   # Unpickling
  b = pickle.load(fp)
"""