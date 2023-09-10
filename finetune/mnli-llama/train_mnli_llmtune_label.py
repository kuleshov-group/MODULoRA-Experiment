import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--weight_path', type=str, help='Path to the weights', required=True)
parser.add_argument('--adapter', type=str, help='Path to store adapter weight', required=True)
parser.add_argument('--mbatch_size', type=int, help='mbatch size for training', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Weight Path:', args.weight_path)
print('Adapter Path: ', args.adapter)
print('Seed: ', args.seed)
print('mbatch_size: ', args.mbatch_size)


import random
import json
import os

# import wandb
import torch
import numpy as np
import bitsandbytes as bnb
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from datasets import load_dataset

# from src.dataset import InstructDataset, ChatDataset
# from src.util.dl import set_random_seed, fix_tokenizer, fix_model
# from src.util.io import read_jsonl

from utils import *
from data_mnli_label import *

from llmtune.executor import load_llm, load_adapter
from llmtune.engine.lora.peft import quant_peft


# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

checkpoint = None
seed = args.seed
train_sample_rate = 1.0
val_sample_rate = 1.0
local_rank = 0
# report_to = "wandb"
output_dir = args.adapter

set_random_seed(seed)
logging.set_verbosity_info()

# with open(config_file, "r") as r:
#     config = json.load(r)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

#deepspeed_config = config.get("deepspeed")



### Training Configuration
#trainer_config = config["trainer"]

MICRO_BATCH_SIZE = args.mbatch_size  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 1e-3  # the Karpathy constant
CUTOFF_LEN = 128  # 128 accounts for about 95% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE= 2000

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

trainer_config = transformers.TrainingArguments(
    per_device_train_batch_size = MICRO_BATCH_SIZE,
    per_device_eval_batch_size = MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.06,
    num_train_epochs=EPOCHS,
    # max_steps = 350,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type = "cosine", ## LoRA original paper uses linear
    fp16=True,
    logging_steps=150,
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=300,
    save_steps=300,
    # report_to=report_to,
    output_dir=output_dir,
    optim = "adamw_torch",
    torch_compile = False,
    save_total_limit=2,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False if ddp else None,
)


# ### Apply LoRA
#
# Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.

target_modules = None
target_modules = ['q_proj', 'v_proj'] # edit with your desired target modules
#lora_config = config.get("lora")
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)

callbacks = [SavePeftModelCallback] if lora_config else []
##no need to use callbacks
callbacks = []

training_args = trainer_config


model_name = "huggyllama/llama-13b"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer = fix_tokenizer(tokenizer)
# tokenizer.save_pretrained(output_dir)

dataset = load_dataset('multi_nli')
train_records = dataset['train']
val_records = dataset['validation_matched']
#random.shuffle(train_records)
print("train_record[0]: ",train_records[0])

model_type = "causal"
templates_path = "llama_lora_mnli_label.json"
only_target_loss = False
mode = "instruct"

llmtune_model_name = args.model_name
llmtune_quantized_weights_path = args.weight_path
llmtune_groupsize = 64

if mode == "instruct":
    max_source_tokens_count = 64 # Changed depending on the dataset
    max_target_tokens_count = 4
    target_field = ""
    source_field = "" #does not matter. (original alpaca-lora paper has additional "input" alongside instruction: instruction-input-output vs. instruction-response)

    train_dataset = InstructDataset(
        train_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=train_sample_rate,
        input_type=model_type,
        templates_path=templates_path,
        target_field=target_field,
        source_field=source_field,
        only_target_loss=only_target_loss
    )

    val_dataset = InstructDataset(
        val_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=val_sample_rate,
        input_type=model_type,
        templates_path=templates_path,
        target_field=target_field,
        source_field=source_field,
        only_target_loss=only_target_loss
    )

    ## Save the model
    dataloader_train = torch.utils.data.DataLoader(train_dataset)
    # torch.save(dataloader_train,'dataloader_train.pth')

    dataloader_val = torch.utils.data.DataLoader(val_dataset)
    # torch.save(dataloader_val,'dataloader_val.pth')

else:
    assert False

if "seq2seq" in model_type:
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

print("INPUT_IDS")
print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
print("MASK")
print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
print("LABELS")
print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])

llm, _ = load_llm(
    llmtune_model_name,
    llmtune_quantized_weights_path,
    llmtune_groupsize
)
model = fix_model(llm, tokenizer, use_resize=False)

# Default model generation params
model.config.num_beams = 5
if mode == "instruct":
    max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count

if not ddp and torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

if lora_config:
    #lora_config = LoraConfig(**lora_config)
    # model = get_peft_model(model, lora_config)
    model = load_adapter(model, lora_config=lora_config)

trainer_class = Trainer ##if not omit_base_model_save else TrainerNoBaseSave
print("Trainer class:", trainer_class)
trainer = trainer_class(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=callbacks,
    data_collator=data_collator,
    # preprocess_logits_for_metrics = preprocess_logits_for_metrics,
)

# with wandb.init(project="llama_ft_samsum", name="llama finetuning run") as run: ## changed the name don't forget
checkpoint_dir = output_dir
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
model.save_pretrained(output_dir)