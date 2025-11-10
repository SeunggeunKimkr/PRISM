import argparse
import json
import ast
import re
import torch
from collections import Counter
from typing import Iterable, List, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


INSTRUCTION = """Write ONLY valid Python code for the task below.
- Output code between <py> and </py> only (no extra text).
- No markdown, no explanations, no comments.
- Do not print or read input. Do not write files or access networks.
- Use only Python standard library imports at the top if needed.
- Define exactly one top-level function {func_name} (keep the name).
- Do not include tests or `if __name__ == "__main__":`.

Task:
{task_text}

Format:
<py>
# optional stdlib imports

def {func_name}(...):
    ...
</py>
"""

def strip_fences(code: str) -> str:
    return re.sub(r"```(?:python)?\s*|\s*```", "", code or "").strip()


def extract_function_from_code(code: str) -> Optional[str]:
    code = strip_fences(code)
    if not code:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        m = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(", code, flags=re.M)
        return m.group(1) if m else None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node.name


def build_prompt(task_text: str, func_name: str) -> str:
    """
    Build the prompt for the task
    """
    return INSTRUCTION.format(task_text=task_text, func_name=func_name)


def load_opc_dataset(dataset_name: str, tokenizer: AutoTokenizer, max_length: int, split_ratio: float = 0.05):
    ds = load_dataset(dataset_name, "educational_instruct")['train']
    ds = ds.train_test_split(test_size=split_ratio)
    train_ds = ds['train']
    test_ds = ds['test']

    def process_sample(sample):
        prompt = build_prompt(sample['instruction'], sample['entry_point'])
        code = "\n" + "<py>"+ "\n" + sample['code'] + "\n" + "</py>"

        prompt_tokens = tokenizer(prompt, add_special_tokens = False)['input_ids']
        code_tokens = tokenizer(code, add_special_tokens = False)['input_ids']
        input_ids = prompt_tokens + code_tokens

        len_without_pad = len(input_ids)

        # pad to max_length
        if len(input_ids) < max_length:
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        elif len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        # boolean tensor for non-prompt tokens
        effective_len = min(len(prompt_tokens), max_length)
        valid_tokens = [(i >= effective_len) for i in range(max_length)]

        return {
            "input_ids": input_ids,
            "valid_tokens": valid_tokens}

        # return {
        #     "input_ids": input_ids,
        #     "valid_tokens": valid_tokens,
        #     "len_without_pad": len_without_pad,
        #     "len_without_prompt": len_without_pad - len(prompt_tokens),
        # }
    
    processed_ds = train_ds.map(process_sample, remove_columns=train_ds.column_names)
    processed_test_ds = test_ds.map(process_sample, remove_columns=test_ds.column_names)

    return processed_ds, processed_test_ds


def mbpp_process():
    args = argparse.ArgumentParser()
    args.add_argument("--out", default = "mbpp_prompts.jsonl")
    args = args.parse_args()

    sanity_sample_path = "sanity_samples.jsonl"

    ds = load_dataset("mbpp", split = "train+validation+test", download_mode = "force_redownload")
    number_of_examples = len(ds)
    print(f"[info] loaded mbpp dataset | size={number_of_examples}")

    with open(args.out , "w" , encoding = "utf-8") as fout, open(sanity_sample_path, "w", encoding = "utf-8") as fsanity:
        for i, ex in enumerate(ds):
            task_id = ex.get("task_id")
            task_text = ex.get("text")
            code = ex.get("code")
            test_list = ex.get("test_list")
            
            # extract the function name and build the prompt
            # for evalution, we also add the test cases
            func_name = extract_function_from_code(code)
            prompt = build_prompt(task_text, func_name)
            rec = {
                "task_id": task_id,
                "prompt": prompt,
                "test_list": test_list,
            }
            sanity_rec = {
                "task_id": task_id,
                "solution": code,
            }
            fout.write(json.dumps(rec, ensure_ascii = False) + "\n")
            fsanity.write(json.dumps(sanity_rec, ensure_ascii = False) + "\n")
            if i % 100 == 0:
                print(f"Processed {i} examples out of {number_of_examples}")
    print(f"Saved {i+1} examples to {args.out}")


def summarize_dataset(ds):
    import numpy as np
    import matplotlib.pyplot as plt

    a = np.array([ex["len_without_pad"] for ex in ds], dtype=np.int64)
    b = np.array([ex["len_without_prompt"] for ex in ds], dtype=np.int64)

    # plot the histogram
    plt.figure()
    plt.hist(a, bins = 100, density = True)
    plt.xlabel("Length without pad")
    plt.ylabel("Density")
    plt.savefig("plots/len_without_pad.png")
    plt.close()

    plt.figure()
    plt.hist(b, bins = 100, density = True)
    plt.xlabel("Length without prompt")
    plt.ylabel("Density")
    plt.savefig("plots/len_without_prompt.png")
    plt.close()
    print(f"Saved histograms to len_without_pad.png and len_without_prompt.png")



if __name__ == "__main__":
    # opc processing
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True)
    max_length = 1024
    opc_ds, opc_test_ds = load_opc_dataset("OpenCoder-LLM/opc-sft-stage2", tokenizer, max_length)
    print(f"Loaded {len(opc_ds)} examples from OpenCoder-LLM/opc-sft-stage2")
    print(f"Loaded {len(opc_test_ds)} examples from OpenCoder-LLM/opc-sft-stage2")

    summarize_dataset(opc_ds)
    summarize_dataset(opc_test_ds)