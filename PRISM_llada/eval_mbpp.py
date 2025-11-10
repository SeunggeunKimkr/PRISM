import os, re, json, argparse, sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from datasets import load_dataset
from evaluate import load as load_metric

# ------------------------------------------------------------
# read a given .json file
# ------------------------------------------------------------

PAIR_STUB = """\
class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b
"""


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def strip_code_fences(text: str) -> str:
    m = re.search(r"<py>\s*(.*?)\s*</py>", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s*(.*?)\s*```", text, re.S | re.I) or re.search(r"```\s*(.*?)\s*```", text, re.S | re.I)
    return (m.group(1).strip() if m else (text or "").strip())

def build_reference_from_tetlist(test_list: List[dict]) -> str:
    ref = "\n".join(test_list or [])
    if "Pair(" in ref and "class Pair" not in ref:
        ref = PAIR_STUB + "\n" + ref
    return ref.strip()

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--prompts", default = "json_samples/mbpp_prompts.jsonl")
    args.add_argument("--samples", default = "json_samples/mbpp_samples.jsonl")
    args.add_argument("--pass_K", default = 1, type = int)
    args = args.parse_args()
    
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')

    # load test cases
    prompts = read_jsonl(args.prompts)
    refs_by_id: Dict[int, str] = {}
    order: List[int] = []
    for r in prompts:
        id = int(r["task_id"])
        test_list = r.get("test_list") or []
        refs_by_id[id] = build_reference_from_tetlist(test_list)
        order.append(id)
    
    order = sorted(set(order))
    
    # load solutions
    solutions_by_id: Dict[int, List[str]] = defaultdict(list)
    samples = read_jsonl(args.samples)
    for s in samples:
        id = int(s["task_id"])
        sol = strip_code_fences(s["solution"])
        solutions_by_id[id].append(sol)
    
    # align the solutions with the test cases
    prediction: List[List[str]] = [solutions_by_id.get(id, []) for id in order]
    reference: List[str] = [refs_by_id[id] for id in order]
    k_list = [args.pass_K]

    # evaluate
    metric = load_metric("code_eval")
    key = f"pass@{args.pass_K}"
    pass1, results = metric.compute(references=reference, predictions=prediction, k=k_list)
    print(f"[RESULT]: {pass1[key]}")

    # collect task_ids that passed
    passed_task_ids = []

    for idx in range(len(order)):
        entries = results.get(idx , [])
        pass1 = bool(entries and entries[0][1].get("passed", False))
        if pass1:
            passed_task_ids.append(order[idx])
    
    print(f"[PASSED TASK IDS]: {passed_task_ids}")

if __name__ == "__main__":
    main()