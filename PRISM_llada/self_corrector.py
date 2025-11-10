import torch
import torch.nn.functional as F
from transformers import Trainer, DefaultDataCollator
from sampling import add_gumbel_noise

SA_WINDOW = 512

# ------------------------------------------------------------------------------------- #
# ------------------------ LLaDA Collator --------------------------------------------- #
# ------------------------------------------------------------------------------------- #

class LLaDACollator(DefaultDataCollator):
    def __init__(self, tokenizer, max_length):
        super().__init__()
        self.mask_token_id = 126336 # LLaDA mask token id
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def forward_process(self , batch , eps = 1e-3):
        x : torch.Tensor = batch["input_ids"]
        valid_tokens = batch.pop("valid_tokens")
        B, L = x.shape

        # here we truncate the valid_tokens (otherwise it would include too many mask tokens)
        cum_sum = valid_tokens.cumsum(dim=-1)
        valid_tokens = valid_tokens & (cum_sum <= 512)

        # the left parts are the same as the standard MDM forward process
        t = (1-eps) * torch.rand(B, device=x.device) + eps
        t = t[:, None].expand(B, L) # (B, L)
        rand = torch.rand((B,L), device=x.device) 
        masked_indices = (rand < t) & valid_tokens
        mask_tokens = torch.full_like(x , self.mask_token_id)
        xt = torch.where(masked_indices, mask_tokens, x)

        return xt, t, masked_indices


    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["x0"] = batch["input_ids"].clone()
        xt , batch["t"], batch["masked_indices"] = self.forward_process(batch)
        batch["input_ids"] = xt.long()
        return batch


class RemaskingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        extra = kwargs.pop("additional_training_args" , {})
        super().__init__(*args, **kwargs)
        self.reg_coeff = extra["reg_coeff"]
        self.K = extra["K"]
        self.max_length = extra["max_length"]
        self.xs_sampling = extra["xs_sampling"]
        self.mask_token_id = 126336 # LLaDA mask token id

    def one_step_sampler(self, logits, xt):
        masked_idx = (xt == self.mask_token_id)
        B = xt.size(0)
        neg_inf = torch.full((B, xt.size(1)), -1e9, device=xt.device)

        # categorical sampling via Gumbel noise--we set temperature = 0.0 as we do in the LLaDA inference
        x0 = torch.argmax(add_gumbel_noise(logits, 0.0), dim=-1)
        if self.xs_sampling == "random":
            # select which positions to unmask--random score
            scores = torch.where(masked_idx, torch.rand(xt.shape, device=xt.device), neg_inf)
    
        elif self.xs_sampling == "semi-autoregressive":
            # select which positions to unmask--make just leftmost SA_WINDOW tokens valid
            # we set the score as the prob_max as we do in the LLaDA inference
            masked_cumsum = masked_idx.to(torch.float32).cumsum(dim=-1)
            valid_mask_indices = masked_idx & (masked_cumsum <= SA_WINDOW)
            p = F.softmax(logits, dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            scores = torch.where(valid_mask_indices, x0_p, neg_inf)

        else:
            raise ValueError(f"Invalid sampling method: {self.xs_sampling}")


        # calculate how many positions to update
        topk_idx = scores.topk(k=self.K , dim = -1).indices # (B, K)
        num_masked = masked_idx.sum(dim=-1) # (B, )
        k_per = torch.minimum(torch.tensor(self.K), num_masked) # (B, )
        take = torch.arange(self.K , device = xt.device).unsqueeze(0) < k_per.unsqueeze(1) # (B, K)

        # update indices--boolean tensor of shape (B, max_length)
        update_ids = torch.zeros(xt.shape, device = xt.device, dtype = torch.bool) # (B, max_length)
        bidx = torch.arange(B, device = xt.device).unsqueeze(1).expand_as(topk_idx) # (B, K)
        update_ids[bidx[take] , topk_idx[take]] = True
        update_ids = update_ids & masked_idx

        xs = xt.clone()
        xs[update_ids] = x0[update_ids]

        return xs, update_ids

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        self-correction loss computation, based on the algorithm in the paper
        also includes regularization loss computation
        """
        batch_size = inputs["input_ids"].size(0)
        t, x0, masked_indices = inputs.pop("t"), inputs.pop("x0"), inputs.pop("masked_indices")
        logits = model(**inputs)["logits"]
        normalize_constant = 512

        with torch.no_grad(): # no need to compute gradient in this step
            xs, update_ids = self.one_step_sampler(logits, inputs["input_ids"])

        # unmasking loss
        unmasking_weight = 1 / t
        unmasking_loss = unmasking_weight[masked_indices] * F.cross_entropy(logits[masked_indices], x0[masked_indices], reduction="none")
        unmasking_loss_scaled = unmasking_loss.sum() / (batch_size * normalize_constant)
        
        # self-correction loss
        binary_label = (xs[update_ids] == x0[update_ids]).float()
        remasking_conf = model(xs)["remasking_conf"]
        self_correction_loss = F.binary_cross_entropy_with_logits(remasking_conf.squeeze(-1)[update_ids], binary_label, reduction="none").sum()
        self_correction_loss_scaled = self_correction_loss / (batch_size * self.K)

        # total loss
        loss = self_correction_loss_scaled + self.reg_coeff * unmasking_loss_scaled

        # sanity check
        print("-"*100)
        print(f"Updated_indices: {update_ids.sum().item()}")
        print(f"Unmasked_indices: {masked_indices.sum().item()}")
        print("-"*100)

        if not hasattr(self, "_ga_idx"):
            self._ga_idx = 0
        if not hasattr(self, "_optim_step"):
            self._optim_step = 0

        # track micro-steps and detect when an optimizer step happens
        self._ga_idx = (self._ga_idx + 1) % self.args.gradient_accumulation_steps
        is_optim_step = (self._ga_idx == 0)
        if is_optim_step:
            self._optim_step += 1

        # log every 10 optimizer steps
        do_log = is_optim_step and (self._optim_step % 10 == 0)

        # IMPORTANT: collective gather must be called by all ranks when do_log is True
        if do_log:
            # detach so logging doesn’t keep the graph
            u_all = self.accelerator.gather_for_metrics(unmasking_loss_scaled.detach())
            sc_all = self.accelerator.gather_for_metrics(self_correction_loss_scaled.detach())
            u_mean = u_all.mean()
            sc_mean = sc_all.mean()

            if self.accelerator.is_main_process:
                self.log({
                    "unmasking_loss": u_mean.item(),
                    "self_correction_loss": sc_mean.item(),
                })

        return loss if not return_outputs else (loss, logits)

    
    
