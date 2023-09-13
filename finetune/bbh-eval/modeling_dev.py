from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import transformers
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel
from torchvision.datasets.utils import download_url
from transformers import AutoTokenizer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
    LlamaConfig,
)

# import quant


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    lora_path: str = ""
    device: str = "cuda"
    load_4bit: bool = False
    load_8bit: bool = False
    load_16bit: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.load_4bit:
                args.update(device_map="auto", load_in_4bit=True)
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            if self.load_16bit:
                args.update(device_map="auto", torch_dtype=torch.float16)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not (self.load_4bit or self.load_8bit or self.load_16bit):
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)

    def get_choice(self, text: str, **kwargs) -> Tuple[int]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        start_token = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                decoder_input_ids=start_token,
                **kwargs,
            ).logits[0, 0]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B

class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, cache_dir="your_cache_path")
        if self.model is None:
            args = {}
            if self.load_4bit:
                args.update(device_map="auto", load_in_4bit=True)
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            if self.load_16bit:
                args.update(device_map="auto", torch_dtype=torch.float16)
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, cache_dir="your_cache_path", **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()

            if not (self.load_4bit or self.load_8bit or self.load_16bit):
                self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            text = template.format_map(dict(instruction=prompt))
        else:
            text = prompt

        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        if "65b" in self.model_path.lower():
            self.max_input_length = 1024
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)

    def get_choice(self, text: str, **kwargs) -> Tuple[int]:
        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def noop(*args, **kwargs):
    assert args is not None
    assert kwargs is not None


def load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    fused_mlp=True,
    warmup_autotune=True,
):
    config = LlamaConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()

    layers = find_layers(model)
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]

    quant.make_quant_linear(model, layers, wbits, groupsize)
    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)

    model.seqlen = 2048
    print("Done.")
    return model


class OPTQModel(LlamaModel):
    quantized_path: str
    tokenizer_path: str
    model: Optional[LlamaForCausalLM]
    tokenizer: Optional[LlamaTokenizer]
    num_bits: int = 4
    group_size: int = 64

    def load(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.model is None:
            from llmtune.executor import load_llm
            if self.quantized_path not in ["llama-7b-4bit", "llama-7b-3bit", "llama-7b-2bit"]:
                self.model, self.tokenizer = load_llm(self.quantized_path, f"your_{self.quantized_path}.pt", groupsize=64)
            else:
                self.model, self.tokenizer = load_llm(self.quantized_path, f"your_{self.quantized_path}.pt", groupsize=64)

            if self.lora_path:
                from llmtune.executor import load_adapter
                if self.lora_path in ['lora-65b-3bit']:
                    self.model = load_adapter(self.model, adapter_path=f"your_{self.lora_path}")
                else:
                    self.model = load_adapter(self.model, adapter_path=f"your_{self.lora_path}")
            self.model.to(self.device)

        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
            self.test_max_length()
        
        from llmtune.utils import to_half_precision
        self.model = to_half_precision(self.model)

    def test_max_length(self):
        # Detect any OOMs at the beginning
        text = " ".join(["test sentence for max length"] * 1000)
        self.run(text)


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        seq_to_seq=SeqToSeqModel,
        llama=LlamaModel,
        optq=OPTQModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


if __name__ == "__main__":
    Fire()
