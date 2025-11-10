import torch
import argparse
import random
import numpy as np
import wandb
import os
from llada import RemaskingLLaDA
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path
from code_utils import load_opc_dataset
from self_corrector import RemaskingTrainer, LLaDACollator


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


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------
# Util function for counting the number of parameters
# ------------------------------------------------------------
def count_parameters(named_params, key: str | None = None):
    return sum(p.numel()
        for n, p in named_params
        if p.requires_grad and (key is None or key in n)
    )

# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # training configuration
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size_device", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Learning rate for the LoRA")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for other parameters")
    parser.add_argument("--grad_accum_steps", type=int, default=3, help="Gradient accumulation steps")

    # self-correction configuration
    parser.add_argument("--reg_coeff", type=float, default=0.5, help="Regularization coefficient")
    parser.add_argument("--K", type=int, default=8, help="Number of tokens to transfer in the self-correction training")
    parser.add_argument("--xs_sampling", type=str, default="semi-autoregressive", help="Sampling method for xs", choices=["semi-autoregressive", "random"])

    # misc configuration
    parser.add_argument("--output_dir", type=str, default="/n/netscratch/sham_lab/Everyone/jay_llada", help="Output directory")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length for tokenization")
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--job_name", type=str, default="jay_debug", help="Job name")

    # dataset configuration
    parser.add_argument("--ft_dataset", type=str, default="OpenCoder-LLM/opc-sft-stage2", help="Dataset for fine-tuning")

    return parser.parse_args()

# Model loading with LoRA integration
def load_model_and_tokenizer(args):
    backbone = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        return_dict=True,
    )
    print("Backbone LLaDA loaded!")
    
    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True

    # add LoRA to backbone first
    lora_config = LoraConfig(
        r=256,
        lora_alpha = 256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.1,
        bias = "none",
        task_type = TaskType.CAUSAL_LM,
    )
    backbone = get_peft_model(backbone, lora_config)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", padding_side="right", trust_remote_code=True, use_fast=True)
    print("tokenizer loaded!")

    model = RemaskingLLaDA(backbone, d_model=4096).to(torch.bfloat16)

    return tokenizer, model


# training setup
def train_model(args, tokenizer, model):    
    # setup the trainable parameters
    lora_params = [p for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "lora" not in n and p.requires_grad]
    
    # parameter count check
    num_lora_params = count_parameters(model.named_parameters(), key = 'lora')
    whole_trainable_params = count_parameters(model.named_parameters())
    print(f"Total trainable parameters: {whole_trainable_params:,}")
    print(f"  └─ LoRA adapter params          : {num_lora_params:,}")
    print(f"  └─ From-scratch params          : {whole_trainable_params - num_lora_params:,}")

    # optimizer setup
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.1},
            {"params": head_params, "lr": args.lr, "weight_decay": 0.1}
        ],
    )

    # load dataset
    train_dataset, test_dataset = load_opc_dataset(args.ft_dataset, tokenizer, args.max_length)
    num_train_steps = int(
        len(train_dataset)
        * args.num_epochs
        / (args.batch_size_device * args.grad_accum_steps * torch.cuda.device_count())
    )

    # training arguments setup
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size_device,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps = 100,
        logging_steps = 10,
        save_steps = 2000,
        save_safetensors=False,
        load_best_model_at_end=False,
        max_grad_norm=1.0,
        bf16=True,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.05,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb else None,
        dataloader_pin_memory=False,
    )

    additional_training_args = {"reg_coeff": args.reg_coeff, "K": args.K, "max_length": args.max_length, "xs_sampling": args.xs_sampling}
    # initialize trainer
    trainer = RemaskingTrainer(
        model = model,
        args = training_args,
        data_collator = LLaDACollator(tokenizer=tokenizer, max_length=args.max_length),
        additional_training_args = additional_training_args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        optimizers = (optimizer, None),
    )

    # train the model
    rank = int(os.environ.get("GLOBAL_RANK", 0))

    if args.wandb and rank == 0:
        wandb.init(project="Remasking-LLaDA", name=args.job_name, entity="jaeyeon_kim-harvard-university")

    # Start training
    trainer.train()

if __name__ == "__main__":
    init_seed(42)
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)