import os, re, json, argparse, sys, ast, importlib, glob
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

# ------------------------------------------------------------
# more thorough evaluation, we also implement L0/L1/L2 eval
# ------------------------------------------------------------

def parse_entry_name_from_prompt(prompt: str):
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", prompt)
    return m.group(1) if m else None

def check_L0(raw_text: str, code: str):
    fenced = bool(re.search(r"<py>.*?</py>", raw_text, re.S|re.I) or
                  re.search(r"```(?:python)?\s.*?```", raw_text, re.S|re.I))
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    if not fenced and raw_text.strip() != code.strip():
        return False, "Code text is mixed with other text"
    return True, "ok"


def check_L1(code: str, expected_name: str | None):
    try:
        t = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # top-level: import / function definition only allowed
    tops = t.body
    for n in tops:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            continue
        return False, f"Unallowed top-level statement: {type(n).__name__}"

    funcs = [n for n in tops if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return False, f"top-level function should be only one (currently {len(funcs)})"
    if expected_name and funcs[0].name != expected_name:
        return False, f"Function name mismatch: expected {expected_name}, actual {funcs[0].name}"

    # banned calls (global and function internal) detection
    banned = {"print", "input", "open"}
    class V(ast.NodeVisitor):
        def __init__(self): self.bad=[]
        def visit_Call(self, node):
            name = None
            if isinstance(node.func, ast.Name): name = node.func.id
            elif isinstance(node.func, ast.Attribute): name = node.func.attr
            if name in banned: self.bad.append(name)
            self.generic_visit(node)
    v = V(); v.visit(t)
    if v.bad:
        return False, f"Banned call used: {sorted(set(v.bad))}"

    # (optional) only standard library is allowed
    not_std = []
    stdnames = getattr(sys, "stdlib_module_names", set())  # Py3.10+
    for n in tops:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import) else [n.module.split('.')[0]]
            for base in mods:
                if stdnames and base not in stdnames:
                    not_std.append(base)
                else:
                    spec = importlib.util.find_spec(base)
                    if spec and spec.origin and "site-packages" in (spec.origin or ""):
                        not_std.append(base)
    if not_std: # if third-party import is used
        return False, f"Third-party import used: {sorted(set(not_std))}"

    return True, "OK"

def evaluate(prompts, samples, pass_K):
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
    # load test cases
    refs_by_id: Dict[int, str] = {}
    order: List[int] = []
    for r in prompts:
        id = int(r["task_id"])
        test_list = r.get("test_list") or []
        refs_by_id[id] = build_reference_from_tetlist(test_list)
        order.append(id)
    
    order = sorted(set(order))

    # extract function names
    name_by_id = {}
    for r in prompts:
        id = int(r["task_id"])
        name_by_id[id] = parse_entry_name_from_prompt(r.get("prompt", ""))
    
    # load solutions
    solutions_by_id: Dict[int, List[str]] = defaultdict(list)
    for s in samples:
        id = int(s["task_id"])
        sol = strip_code_fences(s["solution"])
        solutions_by_id[id].append(sol)
    
    # align the solutions with the test cases
    prediction: List[List[str]] = [solutions_by_id.get(id, []) for id in order]
    reference: List[str] = [refs_by_id[id] for id in order]
    k_list = [pass_K]

    # evaluate
    metric = load_metric("code_eval")
    key = f"pass@{pass_K}"
    pass1, results = metric.compute(references=reference, predictions=prediction, k=k_list)
    print(f"[RESULT]: {pass1[key]}")

    # summarize the results
    summary = []
    for idx, task_id in enumerate(order):
        exp = name_by_id.get(task_id)
        entries = results.get(idx, [])
        for k, code in enumerate(prediction[idx] or []):
            raw = solutions_by_id[idx][k]
            l0, l0_msg = check_L0(raw, code)
            l1, l1_msg = check_L1(code, exp)
            l2_ok = bool(entries and entries[k][1].get("passed", False))
            summary.append( {
                "task_id": task_id,
                "raw": raw,
                "code": code,
                "l0": l0,
                "l0_msg": l0_msg,
                "l1": l1,
                "l1_msg": l1_msg,
                "l2_ok": l2_ok
            })
    sample_name = os.path.basename(samples_file).split(".")[0]
    json_path = f"summary_{sample_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SUMMARY]: {len(summary)} entries saved to {json_path}")

if __name__ == "__main__":
    # for loop for eval
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default = "humaneval_prompts_wo_test_cases.jsonl")
    parser.add_argument("--pass_K", default = 1, type = int)
    args = parser.parse_args()
    
    os.environ.setdefault('HF_ALLOW_CODE_EVAL', '1')
    samples_files = sorted(glob.glob("humaneval_re_*"))
    prompts = read_jsonl(args.prompts)
    if not samples_files:
        raise ValueError("No samples files found")
    for samples_file in samples_files:
        print(f"Evaluating {samples_file}...")
        samples = read_jsonl(samples_file)
        evaluate(prompts, samples, args.pass_K)
        print(f"Evaluation complete for {samples_file}")