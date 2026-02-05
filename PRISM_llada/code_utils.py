import argparse
import json
import ast
import re
import torch
from collections import Counter
from signature_extract import extract_exact_signature_from_prompt
from statistics import mean
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

INSTRUCTION_WITH_SIG = """Write ONLY valid Python code for the task below.
- Output code between <py> and </py> only (no extra text).
- No markdown, no explanations, no comments.
- Do not print or read input. Do not write files or access networks.
- Use only Python standard library imports at the top if needed.
- Define exactly one top-level function matching this signature (do not change it):

{func_signature}

Task:
{task_text}

Format:
<py>
# optional stdlib imports

{func_signature}
    ...
</py>
"""

def strip_fences(code: str) -> str:
    return re.sub(r"```(?:python)?\s*|\s*```", "", code or "").strip()

def strip_doctests(doc: str) -> str:
    """
    Remove doctest-style example blocks from a docstring.

    We treat any line starting with '>>>' (after stripping leading spaces)
    as the start of an example block, and skip that line plus subsequent
    non-blank lines until a blank line is seen.
    """
    lines = doc.splitlines()
    out_lines = []
    in_example = False

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith(">>>"):
            # Start or continue a doctest example block
            in_example = True
            continue

        if in_example:
            # Skip lines that are part of the example block
            if stripped == "":
                # Blank line ends the example block
                in_example = False
            # Either way, don't keep this line
            continue

        # Normal line, keep it
        out_lines.append(line)

    # Remove leading/trailing blank lines
    return "\n".join(out_lines).strip("\n")


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

def build_prompt_with_signature(task_text: str, func_signature: str) -> str:
    """
    Build the prompt for the task with the function signature
    """
    return INSTRUCTION_WITH_SIG.format(task_text=task_text, func_signature=func_signature)

def humaneval_length_summary():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default="rebuttal_eval/humaneval_prompts.jsonl",
                        help="Path to humaneval_prompts.jsonl")
    parser.add_argument("--add_special_tokens", action="store_true",
                        help="Include model special tokens when tokenizing")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        padding_side="right",
        trust_remote_code=True,
        use_fast=True,
    )

    lengths = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            prompt = ex["prompt"]
            enc = tokenizer(
                prompt,
                add_special_tokens=args.add_special_tokens,
            )
            lengths.append(len(enc["input_ids"]))

    lengths.sort()
    n = len(lengths)
    if n == 0:
        print("No examples found.")
        return

    def pct(p):
        idx = int(p * (n - 1))
        return lengths[idx]

    print(f"# examples: {n}")
    print(f"min length: {lengths[0]}")
    print(f"max length: {lengths[-1]}")
    print(f"mean length: {mean(lengths):.2f}")
    print(f"median (p50): {pct(0.50)}")
    print(f"p75: {pct(0.75)}")
    print(f"p90: {pct(0.90)}")
    print(f"p95: {pct(0.95)}")
    print(f"p99: {pct(0.99)}")


def extract_docstring_from_prompt(prompt: str) -> Optional[str]:
    """
    Parse the HumanEval prompt and return the function's docstring
    """
    source = prompt
    if not source.endswith("\n"):
        source += "\n"
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return ast.get_docstring(node)
    return None
    
def build_humaneval_task_text(prompt: str) -> str:
    """
    Prefer the natural-language docstring as task_text.
    Fall back to the raw prompt code if there is no docstring.
    """
    doc = extract_docstring_from_prompt(prompt)
    if doc:
        return doc.strip()
    return prompt.strip()


def build_humaneval_task_text_wo_examples(prompt: str) -> str:
    """
    Use the natural-language docstring, but strip doctest-style examples.
    Fall back to the raw prompt code if there is no docstring.
    """
    doc = extract_docstring_from_prompt(prompt)
    if doc:
        cleaned = strip_doctests(doc)
        # If stripping examples leaves nothing, fall back to the original doc
        return (cleaned or doc).strip()
    return prompt.strip()


def extract_test_list(test_str: str, entry_point: str):
    """
    Convert HumanEval's `test` string (check(candidate) function)
    into a list of assert strings.

    - Extract each `assert ...` block.
    - Replace 'candidate' with the actual entry_point function name.
    """
    # Grab each assert block up to the next assert or end of string
    asserts_raw = re.findall(
        r"assert.*?(?=\n\s*assert|$)",
        test_str,
        flags=re.DOTALL,
    )
    test_list = []
    for a in asserts_raw:
        s = a.replace("candidate", entry_point).strip()
        if s:
            test_list.append(s)

    # Fallback: if we somehow got nothing, keep the whole test code
    if not test_list and test_str.strip():
        test_list = [test_str.strip().replace("candidate", entry_point)]

    return test_list


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


def normalize_task_id(raw_id) -> int:
    """
    Convert HumanEval's task_id (e.g. 'HumanEval/0', 'test/21') to an int.
    """
    if isinstance(raw_id, int):
        return raw_id
    s = str(raw_id)
    m = re.search(r"(\d+)$", s)
    if not m:
        raise ValueError(f"Cannot extract integer task_id from {raw_id!r}")
    return int(m.group(1))


def humaneval_process(include_test_cases: bool = True, with_signature: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="humaneval_prompts.jsonl")
    parser.add_argument("--with_signature", action="store_true",
                        help="Include the function signature in the prompt")
    args = parser.parse_args()
    if not include_test_cases:
        out_path = args.out.replace(".jsonl", "_wo_test_cases.jsonl")
    if with_signature:
        out_path = args.out.replace(".jsonl", "_with_signature.jsonl")
    else:
        out_path = args.out

    ds = load_dataset("openai_humaneval", split="test")
    num_examples = len(ds)
    print(f"[info] loaded openai_humaneval | size={num_examples}")

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, ex in enumerate(ds):
            raw_task_id     = ex["task_id"]
            task_id = normalize_task_id(raw_task_id)
            prompt_code = ex["prompt"]
            test_code   = ex["test"]
            entry_point = ex["entry_point"]

            # Build task_text (docstring preferred, else raw prompt)
            if include_test_cases:
                task_text = build_humaneval_task_text(prompt_code)
            else:
                task_text = build_humaneval_task_text_wo_examples(prompt_code)

            if with_signature:
                func_signature = extract_exact_signature_from_prompt(prompt_code, entry_point)
                if not func_signature:
                    func_signature = f"def {entry_point}(...):"
                prompt = build_prompt_with_signature(task_text, func_signature)
            else:
                prompt = build_prompt(task_text, entry_point)

            # Build test_list from HumanEval's test field
            test_list = extract_test_list(test_code, entry_point)

            rec = {
                "task_id": task_id,
                "prompt": prompt,
                "test_list": test_list,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"Processed {i} / {num_examples} HumanEval examples")

    print(f"Saved {num_examples} examples to {out_path}")


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
    # tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True)
    # max_length = 1024
    # opc_ds, opc_test_ds = load_opc_dataset("OpenCoder-LLM/opc-sft-stage2", tokenizer, max_length)
    # print(f"Loaded {len(opc_ds)} examples from OpenCoder-LLM/opc-sft-stage2")
    # print(f"Loaded {len(opc_test_ds)} examples from OpenCoder-LLM/opc-sft-stage2")

    # summarize_dataset(opc_ds)
    # summarize_dataset(opc_test_ds)


    # humeaneval proprocessing
    # humaneval_process()

    humaneval_process(include_test_cases=False, with_signature=True)