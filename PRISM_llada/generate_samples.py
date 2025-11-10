import argparse, json, os, random, re
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from llada import RemaskingLLaDA
from sampling import llada_inference
from pathlib import Path

# ---------------- utils ----------------
def init_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def build_output_prefix(args) -> str:
    prefix = f"{args.output_path}_{args.model_type}_{args.remasking_mode}_{args.steps}"
    if args.remasking:
        prefix += f"_remasking_{args.num_remasking}_pass{args.pass_K}"
    else:
        prefix += f"_no_remasking_pass{args.pass_K}"
    return prefix

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            rows = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows

def extract_solution(code: str) -> str:
    # extract the code between the last <py> and </py>
    blocks = re.findall(r'(?is)<py>\s*(.*?)\s*</py>', code)
    return blocks[-1].strip() if blocks else code.strip()

def find_prompt(rows: List[Dict[str, Any]], task_id: int) -> str:
    for r in rows:
        if r["task_id"] == task_id:
            return r["prompt"]
    raise ValueError(f"Task id {task_id} not found")

# ---------------- dataset ----------------
class PromptDataset(Dataset):
    def __init__(self, recs: List[Dict[str, Any]]):
        self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, idx):
        r = self.recs[idx]
        return {"task_id": int(r["task_id"]), "prompt": r["prompt"]}

def collate(batch):
    return {
        "task_id": [b["task_id"] for b in batch],
        "prompt":  [b["prompt"]  for b in batch],
    }

# ---------------- generation ----------------
@torch.no_grad()
def generate_samples(model, tokenizer, loader, args, device, rank) -> int:
    model.eval()
    output_path = build_output_prefix(args)
    shard_path = f"{output_path}.rank{rank}"

    written = 0
    with open(shard_path, "w", encoding="utf-8") as fout:
        pbar = tqdm(loader, disable=(rank != 0), desc=f"rank{rank}")
        for batch in pbar:
            task_ids = batch["task_id"]
            prompts  = batch["prompt"]

            enc = tokenizer(
                prompts,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            for k in range(int(args.pass_K)):
                # define a new seed
                seed_k = args.seed + 1000 * k + rank
                init_seed(seed_k)
                out = llada_inference(
                    model, enc["input_ids"], tokenizer, args.steps, args.max_length, args.model_type, args.block_length,
                    args.unmasking, args.remasking, args.remasking_mode, args.num_remasking, args.temperature
                )
                codes = tokenizer.batch_decode(out, skip_special_tokens=True)
            
                for tid, code in zip(task_ids, codes):
                    solution = extract_solution(code)
                    rec = {"task_id": int(tid), "solution": solution}
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

            if rank == 0:
                pbar.set_postfix_str(f"last_batch={len(task_ids)}")

    return written

def merge_shards(out_path: str, world_size: int):
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in range(world_size):
            shard = f"{out_path}.rank{r}"
            if not os.path.exists(shard): continue
            with open(shard, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(shard)


def PRISM_model_load(model_path: str, backbone: AutoModel, device: torch.device):
    # follow the same configurations as in the training script
    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True
    lora_config = LoraConfig(
        r=256,
        lora_alpha = 256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )
    backbone = get_peft_model(backbone, lora_config)
    model = RemaskingLLaDA(backbone, d_model=4096)
    
    # load a model from the checkpoint
    ckpt_dir = Path(model_path)
    state = torch.load(ckpt_dir/"pytorch_model.bin", map_location="cpu")

    # double check the keys
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    # move to the device
    model = model.to(device = device, dtype = torch.bfloat16).to(device)
    print("PRISM model loaded!")

    return model

# ---------------- main ----------------
def main(args):
    init_seed(args.seed)
    rank, world_size = setup_ddp()
    device = torch.device("cuda", rank)
    if rank == 0:
        print(f"[DDP] world_size={world_size}")

    # load prompts
    rows = read_json_or_jsonl(args.prompts)
    dataset = PromptDataset(rows)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate, num_workers=0, pin_memory=True)

    # load model/tokenizer
    backbone = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True)

    if args.model_type == "PRISM":
        model = PRISM_model_load(args.model_path, backbone, device)
    elif args.model_type == "baseline":
        model = backbone

    if rank == 0: print("LLaDA and tokenizer loaded!")

    # generate
    if args.test:
        model.eval()
        prompt = find_prompt(rows, args.task_id)
        print(f"Prompt: {prompt}")
        print("-"*100)
        enc = tokenizer(
            prompt,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = llada_inference(
                model, enc["input_ids"], tokenizer, args.steps, args.max_length, args.model_type, args.block_length,
                args.unmasking, args.remasking, args.remasking_mode, args.num_remasking, args.temperature
        )
        code = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        solution = extract_solution(code)
        print(solution)
        print("-"*100)
        print("Evaluating the code right away...")
        # evaluate the code right away
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
        from evaluate import load as load_metric
        test_list = []
        for r in rows:
            try:
                if int(r.get("task_id", -1)) == int(args.task_id):
                    test_list = r.get("test_list") or []
                    break
            except:
                pass

        reference = "\n".join(test_list).strip()
        if "Pair(" in reference and "class Pair" not in reference:
            pair_stub = (
                "class Pair:\n"
                "    def __init__(self, a, b):\n"
                "        self.a = a\n"
                "        self.b = b\n"
            )
            reference = pair_stub + "\n" + reference

        # Prepare inputs for code_eval
        metric = load_metric("code_eval")
        references = [reference]        # one task
        predictions = [[solution]]      # k=1 prediction for that task

        passk, results = metric.compute(
            references=references,
            predictions=predictions,
            k=[1],
        )

        # results is {0: [(prediction_string, { ... per-run info ... })]}
        entries = results.get(0, [])
        passed = bool(entries and entries[0][1].get("passed", False))

        print(f"[EVAL] task_id={args.task_id}  pass@1={passk['pass@1']:.4f}  passed={passed}")

    else:
        local_written = generate_samples(model, tokenizer, loader, args, device, rank)
        # sync + merge
        dist.barrier()
        if rank == 0:
            output_path = build_output_prefix(args)
            merge_shards(output_path, world_size)
            print(f"[OK] merged -> {output_path}")
        cleanup_ddp()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # model configs
    ap.add_argument("--model_path", default="/n/netscratch/sham_lab/Everyone/jay_llada/llada_PRISM/checkpoint-44000")
    ap.add_argument("--model_type", choices=["baseline", "PRISM"])
    ap.add_argument("--batch_size", type=int, default=4)

    # inference configs
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--unmasking", type=str, default="prob_max")
    ap.add_argument("--remasking", action="store_true")
    ap.add_argument("--block_length", type=int, default=64)
    ap.add_argument("--num_remasking", type=int, default=12, help="number of remasking tokens for each block")
    ap.add_argument("--remasking_mode", type=str, choices=["remdm", "remdm_conf", "PRISM"])
    ap.add_argument("--temperature", type=float, default=0.0)

    # misc configs
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--prompts", default="json_samples/mbpp_prompts.jsonl")
    ap.add_argument("--output_path", default="mbpp_samples")
    ap.add_argument("--test", action="store_true", help="test mode")
    ap.add_argument("--track", action="store_true", help="track the samples")
    ap.add_argument("--task_id", type=int, help="task id to track")
    ap.add_argument("--pass_K", type=int, default=1, help="pass@K")


    args = ap.parse_args()
    main(args)
