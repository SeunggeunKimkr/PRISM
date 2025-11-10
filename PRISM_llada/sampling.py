import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_tokens(token_nums, steps):
    base = token_nums // steps
    remainder = token_nums % steps 

    # create the tensor and then modify in-place
    num_tokens = base.unsqueeze(1).expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=token_nums.device)
        mask = indices.unsqueeze(0) < remainder.unsqueeze(-1)
        num_tokens[mask] += 1

    return num_tokens.to(torch.int64)

def get_num_tokens_centered(token_nums, steps):
    base = token_nums // steps
    remainder = token_nums % steps
    
    # create the tensor and then modify in-place
    num_tokens = base.unsqueeze(1).expand(-1, steps).clone()
    if remainder.sum() > 0:
        device = token_nums.device
        idx = torch.arange(steps, device=device, dtype = torch.float32)
        center = (steps - 1) / 2
        order = torch.argsort(torch.abs(idx - center), stable = True)
        inv_rank = torch.empty(int(steps) , device = device, dtype = torch.int64)
        inv_rank[order] = torch.arange(steps, device = device)

        mask = inv_rank.unsqueeze(0) < remainder.unsqueeze(-1)
        num_tokens[mask] += 1

    return num_tokens.to(torch.int64)


def token_preallocation(block_mask_index, steps_per_block, num_remasking = None):
    if num_remasking is None:
        return get_num_tokens(block_mask_index.sum(dim=-1), steps_per_block)
    else:
        B = block_mask_index.shape[0]
        num_remasking_tensor = num_remasking * torch.ones(B, device=block_mask_index.device, dtype=torch.int64)
        num_unmasking_tokens = get_num_tokens(block_mask_index.sum(dim=-1), steps_per_block)
        num_remasking_tokens = get_num_tokens_centered(num_remasking_tensor, steps_per_block)
        return num_unmasking_tokens + num_remasking_tokens, num_remasking_tokens

def log_trace(x, tokenizer, end_idx, prompt_len, step):
    # helper function to log the trace of the x
    path = f"trace.txt"
    with open(path, "a") as f:
        ids = x[0, prompt_len[0].item() : end_idx[0].item()].detach().cpu().tolist()
        string = tokenizer.decode(ids)
        f.write("-"*100 + "\n")
        f.write(f"{step}: ")
        f.write(string + "\n")
        f.write("-"*100 + "\n")


@torch.no_grad()
def llada_inference(
    model,
    prompt: torch.Tensor,
    tokenizer,
    steps: int,
    max_length: int,
    model_type: str,
    block_length: int,
    unmasking: str,
    remasking,
    remasking_mode: str,
    num_remasking: int,
    temperature: float,
    mask_id: int = 126336,
    track: bool = False,
):
    # Calculate the number of blocks
    assert max_length % block_length == 0, "max_length must be divisible by block_length"
    num_blocks = max_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    print(f"steps_per_block: {steps_per_block}")
    print("--------------------------------")

    # Setup
    if prompt is None:
        raise ValueError("prompt must be provided")
    B = prompt.shape[0]
    if prompt.shape[1] != max_length:
        raise ValueError(f"Prompt length {prompt.shape[1]} does not match max_length {max_length}")

    device = next(model.parameters()).device if hasattr(model, "parameters") else "cuda"
    prompt = prompt.to(device)

    # Initialize with [MASK] and then overwrite with non-pad prompt tokens
    x = torch.full((B, max_length), mask_id, dtype=torch.long, device=device)
    x = torch.where(prompt != tokenizer.pad_token_id, prompt, x)

    pos = torch.arange(max_length, device=device).unsqueeze(0).expand(B, -1)

    # get the prompt index and length per batch
    prompt_index = (x != mask_id)
    prompt_len = prompt_index.sum(dim=1)  # (B,)

    # rank for tqdm gating in distributed setting
    rank = dist.get_rank() if dist.is_initialized() else 0

    # pre-build the neg_inf tensor
    neg_inf = torch.tensor(-np.inf, device=device)

    # Mixed precision
    autocast_device = "cuda" if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda") else "cpu"

    # for remdm_conf, define a buffer-style tensor to store the remasking conf
    if remasking_mode == "remdm_conf":
        # we keep updating the remasking score as sampling proceeds
        remasking_score = torch.zeros((B, max_length), device=device)

    with torch.autocast(device_type=autocast_device, dtype=torch.float16 if autocast_device == "cuda" else torch.bfloat16):
        # Semi-autoregressive sampling over blocks
        for num_block in tqdm(range(num_blocks), disable=(rank != 0), desc=f"Rank {rank}"):
            # Compute block range per batch
            start_idx = (prompt_len + num_block * block_length).to(torch.long)
            end_idx = (start_idx + block_length).clamp(max=max_length)
            # Block mask (B, T)
            block = (pos >= start_idx[:, None]) & (pos < end_idx[:, None])

            # If no masked tokens in block, skip
            block_mask_index = (x == mask_id) & block
            if block_mask_index.sum() == 0:
                continue

            # Pre-allocate the number of unmasking and remasking tokens, size: (B, steps_per_block)
            if remasking:
                num_unmasking_tokens, num_remasking_tokens = token_preallocation(block_mask_index, steps_per_block, num_remasking)
            else:
                num_unmasking_tokens = token_preallocation(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                # Forward pass
                out = model(x)

                # Identify clean tokens within the current block
                block_clean_index = (x != mask_id) & block  # (B, T)
                block_mask_index = (x == mask_id) & block  # (B, T)
                if block_mask_index.sum() == 0:
                    # Nothing to unmask anymore in this block, move to the next block
                    break

                # --------------------- Remasking step ---------------------
                if remasking:
                    if remasking_mode == "remdm":
                        # remdm uses random noise as the remasking score
                        remasking_score = torch.rand(out["remasking_conf"].squeeze(-1).shape, device=out["remasking_conf"].device)
                    else:
                        # low remasking confidence is more likely to be remasked
                        remasking_score = -1.0 * out["remasking_conf"].squeeze(-1)
                    remasking_score = torch.where(block_clean_index , remasking_score, neg_inf)
                    for j in range(remasking_score.shape[0]):
                        k = min(num_remasking_tokens[j, i].item(), int(block_clean_index[j].sum().item()))
                        if k > 0:
                            _, select_indices = torch.topk(remasking_score[j], k=k)
                            x[j, select_indices] = mask_id
                    
                    if track:
                        log_trace(x, tokenizer, end_idx, prompt_len, step = "remasking")
                 
                
                # --------------------- Unmasking step ---------------------

                logits = out["logits"] if model_type == "PRISM" else out.logits
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, T)

                # Probability for the chosen token (for "prob_max" strategy)
                if unmasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                elif unmasking == "prob_max":
                    p = F.softmax(logits, dim=-1)  # (B, T, V)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L)
                else:
                    raise NotImplementedError(f"unmasking strategy '{unmasking}' not supported")

                # Confidence: only valid inside masked positions within the current block
                unmasking_score = torch.where(block_mask_index, x0_p, neg_inf)  # (B, L)

                # Update masked tokens by selecting top-k per batch this step
                for j in range(unmasking_score.shape[0]):
                    k = min(num_unmasking_tokens[j, i].item(), int(block_mask_index[j].sum().item()))
                    if k > 0:
                        _, select_indices = torch.topk(unmasking_score[j], k=k)
                        x[j, select_indices] = x0[j, select_indices]

                        if remasking_mode == "remdm_conf":
                            remasking_score = remasking_score.to(dtype=x0_p.dtype)
                            # update the remasking score as the probability of the chosen token
                            remasking_score[j , select_indices] = x0_p[j, select_indices]
                
                if track:
                    log_trace(x, tokenizer, end_idx, prompt_len, step = "unmasking")

        return x
