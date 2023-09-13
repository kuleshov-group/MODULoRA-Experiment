import os
import json
from argparse import Namespace
from typing import List

from datasets import load_dataset, get_dataset_config_names
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from modeling_dev import select_model, EvalModel


class BBHSample(BaseModel):
    input: str
    target: str

    def as_prompt(self, include_answer: bool = True):
        prompt = self.input
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(self.target)
        return prompt


class BBHData(BaseModel):
    samples: List[BBHSample]

    @classmethod
    def get_config_names(cls, path: str = "lukaemon/bbh") -> List[str]:
        return get_dataset_config_names(path)

    @classmethod
    def load_from_huggingface(
        cls, path: str = "lukaemon/bbh", config: str = "", split: str = "test"
    ):
        data = load_dataset(path, config, split=split)
        samples = [BBHSample(**raw) for raw in tqdm(data, desc=str((path, split)))]
        return cls(samples=samples)


def gen_prompt(data: BBHData, k=-1):
    prompt = ""
    if k == -1:
        k = len(data.samples)
    for i in range(k):
        prompt += data.samples[i].as_prompt()
    return prompt


def evaluate(model: EvalModel, data: BBHData, ntrain: int) -> dict:
    data_train = BBHData(samples=data.samples[:ntrain])
    data_test = BBHData(samples=data.samples[ntrain:])
    is_correct = []

    for i in range(len(data_test.samples)):
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = data_test.samples[i].as_prompt(include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = data_test.samples[i].target
        pred = model.run(prompt)
        is_correct.append(pred.strip().startswith(label))
        if i == 0:
            print(dict(prompt=prompt, label=label, pred=pred))

    return dict(score=sum(is_correct) / len(is_correct))


def main(data_dir: str = "lukaemon/bbh", ntrain: int = 3, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=32, **kwargs)
    print(locals())

    if 'load_4bit' in kwargs:
        loadin_4bit = 'true'
    else:
        loadin_4bit = 'false'

    if 'load_8bit' in kwargs:
        loadin_8bit = 'true'
    else:
        loadin_8bit = 'false'

    if 'lora_path' in kwargs:
        file_name = f"all_results_{kwargs['model_path'].replace('/', '-')}_{kwargs['lora_path'].replace('/', '-')}_4bit_{loadin_4bit}_8bit_{loadin_8bit}.txt"
    else:
        file_name = f"all_results_{kwargs['model_path'].replace('/', '-')}_4bit_{loadin_4bit}_8bit_{loadin_8bit}.txt"

    all_results = []
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            print(f"Loading {file_name}")
            all_results = json.load(f)
            print(all_results)

    start = len(all_results)
    for name in tqdm(BBHData.get_config_names()[start:]):
        data = BBHData.load_from_huggingface(config=name)
        result = evaluate(model, data, ntrain=ntrain)
        all_results.append(result)
        print(dict(name=name, **result))

        # Save the state of all_results after each iteration
        with open(file_name, "w") as f:
            json.dump(all_results, f)

    score = sum(res["score"] for res in all_results) / len(all_results)
    print(dict(average=score))
    return score


if __name__ == "__main__":
    Fire()
