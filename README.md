# MODULoRA
Code repository for the paper "ModuLoRA: Finetuning 3-Bit LLMs on Consumer GPUs by Integrating with Modular Quantizers".

**This repo builds on LLMTune dev branch, with added support of custom dataset preparation and evaluation.**

**Abstract:** We propose a memory-efficient finetuning algorithm for large language models (LLMs) that supports
finetuning LLMs with 65B parameters in 3-bit or 4-bit precision on as little as one 48GB GPU. Our
method, modular low-rank adaptation (MODULORA), integrates any user-specified weight quantizer
with finetuning via low-rank adapters (LoRAs). Our approach relies on a simple quantization-agnostic
backward pass that adaptively materializes low-precision LLM weights from a custom black-box
quantization module. This approach enables finetuning 3-bit LLMs for the first time—leveraging
state-of-the-art 3-bit OPTQ quantization often outperforms finetuning that relies on less sophisticated
4-bit and 8-bit methods. In our experiments, MODULORA attains competitive performance on text
classification, natural language infernece, and instruction following tasks using significantly less
memory than existing approaches, and we also surpass the state-of-the-art ROUGE score on a popular
summarization task. We release MODULORA together with a series of low-precision models—
including the first family of 3-bit instruction following Alpaca LLMs—as part of LLMTOOLS, a
user-friendly library for quantizing, running, and finetuning LLMs on consumer GPU.


# Repository Overview

There are several directories in this repo:
* [llmtune/](llmtune) contains the source code for the package `llmtune`, which needs to be installed to run the examples we provide;
* [examples/](examples/) contains an example implementation of 4-bit, 3-bit quantization using OPTQ, finetuning with alpaca dataset, and model generation after applying finetuned LoRA adapater werights.
* [finetune/samsum-llama/](examples/samsum-llama) contains implementation of finetuning SAMSum benchmark with LoRA in LLaMA models using our package and bitsandbytes, which can be used to reproduce the result in our paper;
* [finetune/mnli-samsum/](examples/mnli-llama) contains implementation of finetuning MNLI benchmark with LoRA in LLaMA models using our package and bitsandbytes, which produces competitive results compared to SOTA;
* Others finetuning scripts can also be found in the same directory [OPT](examples/samsum-opt), [BLOOM](examples/mnli-bloom)
* See how we train `loralib` in [GPT-2](examples/NLG/src/model.py), [RoBERTa](examples/NLU/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa v2](examples/NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py)
* See how we evaluate `loralib` in [GPT-2](examples/NLG/src/model.py), [RoBERTa](examples/NLU/src/transformers/models/roberta/modeling_roberta.py), and [DeBERTa v2](examples/NLU/src/transformers/models/deberta_v2/modeling_deberta_v2.py)

