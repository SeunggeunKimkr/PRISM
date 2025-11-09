import torch
import torch.nn.functional as F
import numpy as np
import utils
from typing import Optional
from dataclasses import dataclass
from prob_path import MaskProbPath
from models.model_wrapper import DITWrapper, FeatureDITWrapper

@dataclass
class Loss:
  loss: torch.FloatTensor
  token_mask: torch.FloatTensor

class BaseLoss:
  def __init__(self, *,
               config,
               mask_index: int,
               vocab_size: int,
               tokenizer,
               noise,
               backbone,
               adapter=None,
               tune_bb: bool=False,
               prob_path: Optional[MaskProbPath]=None):
    self.config = config
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    self.sampling_eps = self.config.training.sampling_eps
    self.adapter_sampling_eps = self.config.adapter_training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.tokenizer = tokenizer
    self.parameterization = self.config.parameterization
    self.mask_index = mask_index
    self.vocab_size = vocab_size
    self.tune_bb = tune_bb
    self.noise = noise
    self.wrapped_backbone= DITWrapper(model = backbone,
                                      time_conditioning = self.time_conditioning,
                                      is_adapter = False)
    self.wrapped_adapter = None
    if adapter is not None:
      if self.config.adapter.input_type == 'embedding':
        self.wrapped_adapter = FeatureDITWrapper(model = adapter)
      else:
        self.wrapped_adapter = DITWrapper(model = adapter,
                                          time_conditioning = self.time_conditioning,
                                          is_adapter = True)
    self.prob_path = prob_path


  def _sample_xt_x0(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    mask_indices = (xt == self.mask_index)

    return xt, mask_indices

  def _sample_t(self, n, device, _eps_0=0.001):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - _eps_0) * _eps_t + _eps_0
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

class PretrainCrossEntropyLoss(BaseLoss):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def __call__(self, x0, attention_mask):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    t = self._sample_t(input_tokens.shape[0], input_tokens.device, _eps_0=self.sampling_eps)
    sigma, dsigma = self.noise(t)
    unet_conditioning = sigma[:, None]
    move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise
    xt, mask_indices = self._sample_xt_x0(input_tokens, move_chance)
    
    # model forward pass
    model_output = self.wrapped_backbone(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    # compute the loss--the reweighting term = 1/t for LogLinearNoise
    reweighting = 1 / t.unsqueeze(-1).expand(-1, input_tokens.size(1))
    ce_loss = F.cross_entropy(model_output[mask_indices], input_tokens[mask_indices], reduction='none')
    ce_loss = ce_loss * reweighting[mask_indices]
    ce_loss_with_attn = ce_loss * attention_mask[mask_indices]
    loss = ce_loss_with_attn.sum() / (input_tokens.shape[0] * input_tokens.shape[1])

    return Loss(loss=loss,
                token_mask=attention_mask)

############################################################
# Adapter Fine-tuning Loss
############################################################
class AdapterFinetuneTokenCriticLoss(BaseLoss):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.wrapped_adapter is not None
    assert self.prob_path is not None
  def __call__(self, x0, attention_mask):
    (input_tokens, output_tokens, 
     attention_mask) = self._maybe_sub_sample(
      x0, attention_mask)
    
    t = self._sample_t(input_tokens.shape[0], input_tokens.device, _eps_0=self.adapter_sampling_eps)
    sigma, dsigma = self.noise(t)
    unet_conditioning = sigma[:, None]
    move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise
    xt, mask_indices = self._sample_xt_x0(input_tokens, move_chance)
    
    num_clean_tokens = self.config.model.length * torch.ones(xt.shape[0], device=xt.device, dtype=torch.long)  
    x0_theta = self.prob_path._topk_static_update(xt, t, k=num_clean_tokens, mode='adapter_finetune')[0].detach()
    adapter_output = self.wrapped_adapter(x0_theta, unet_conditioning)
    utils.print_nans(adapter_output, 'adapter_output')
    reweighting = 1 / t.unsqueeze(-1).expand(-1, input_tokens.size(1))
    binary_labels = (x0_theta == input_tokens).to(adapter_output.device, dtype=adapter_output.dtype)
    bce_loss = F.binary_cross_entropy_with_logits(adapter_output[mask_indices], binary_labels[mask_indices], reduction='none')
    bce_loss = bce_loss * reweighting[mask_indices]
    loss_with_attn = bce_loss * attention_mask[mask_indices]
    loss = loss_with_attn.sum() / (input_tokens.shape[0] * input_tokens.shape[1])
    return Loss(loss=loss,
                token_mask=attention_mask)

class AdapterFinetunePRISMLoss(BaseLoss):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert self.wrapped_adapter is not None
    assert self.prob_path is not None
  def __call__(self, x0, attention_mask):
    bs = x0.shape[0]
    (input_tokens, output_tokens, 
     attention_mask) = self._maybe_sub_sample(
      x0, attention_mask)
    
    t = self._sample_t(input_tokens.shape[0], input_tokens.device, _eps_0=self.adapter_sampling_eps)
    sigma, dsigma = self.noise(t)
    unet_conditioning = sigma[:, None]
    move_chance = 1 - torch.exp(-sigma[:, None]) # (1-eps) * t for LogLinearNoise
    xt, mask_indices = self._sample_xt_x0(input_tokens, move_chance)
    
    #xt -> xs but for different t (k)
    num_clean_tokens = (xt != self.mask_index).sum(dim=-1)
    next_num_clean_tokens = torch.clamp(num_clean_tokens + self.config.adapter_training.num_demasking_tokens_per_step, max=self.config.model.length)
    xs, x0_logits = self.prob_path._topk_static_update(xt, t, k=next_num_clean_tokens, mode='adapter_finetune')
    xs = xs.detach()
    num_xs_xt = self.config.sampling.num_x0_xt * self.config.sampling.num_xs_x0
    if num_xs_xt > 1:
      xt = xt.repeat(num_xs_xt, 1)
      unet_conditioning = unet_conditioning.repeat(num_xs_xt, 1)
      input_tokens = input_tokens.repeat(num_xs_xt, 1)
      attention_mask = attention_mask.repeat(num_xs_xt, 1)
      mask_indices = mask_indices.repeat(num_xs_xt, 1)
      t = t.repeat(num_xs_xt)
    updated_indices = (xs != xt).to(torch.bool)
    if self.config.adapter.input_type == 'xt':
      adapter_output = self.wrapped_adapter(xs, unet_conditioning)
    else: # self.config.adapter.input_type == 'embedding'
      if not self.tune_bb:
        self.wrapped_backbone.eval()
        _, hidden_states_eval, c = self.wrapped_backbone.get_output_and_features(xs, unet_conditioning)
        self.wrapped_backbone.train()
        hidden_states_eval = hidden_states_eval.detach()
        c = c.detach()
      else:
        _, hidden_states_eval, c = self.wrapped_backbone.get_output_and_features(xs, unet_conditioning)
      adapter_output = self.wrapped_adapter(hidden_states_eval, c)
    utils.print_nans(adapter_output, 'adapter_output')
    binary_labels = (input_tokens[updated_indices] == xs[updated_indices]).to(adapter_output.device, dtype=adapter_output.dtype)
    bce_loss = F.binary_cross_entropy_with_logits(adapter_output[updated_indices], binary_labels, reduction='none')
    loss_with_attn = bce_loss * attention_mask[updated_indices]
    mask_reweighting = 1 / t.unsqueeze(-1).expand(-1, input_tokens.size(1))
    clean_reweighting = 1 / (1-t).unsqueeze(-1).expand(-1, input_tokens.size(1))
    if self.config.adapter_training.reweight_loss:
      loss_with_attn = loss_with_attn * clean_reweighting[updated_indices]
      loss = loss_with_attn.sum() / (input_tokens.shape[0] * input_tokens.shape[1])
    else:
      loss = loss_with_attn.sum() / updated_indices.sum() # updated_indices.sum()=k*num_xs_xt*B
    if self.config.adapter_training.reg_lambda > 0:
      input_tokens = input_tokens[:bs]
      attention_mask = attention_mask[:bs]
      mask_indices = mask_indices[:bs]
      mask_reweighting = mask_reweighting[:bs]
      ce_loss = F.cross_entropy(x0_logits[mask_indices], input_tokens[mask_indices], reduction='none')
      ce_loss = ce_loss * mask_reweighting[mask_indices]
      ce_loss_with_attn = ce_loss * attention_mask[mask_indices]
      loss += ce_loss_with_attn.sum() * self.config.adapter_training.reg_lambda / (input_tokens.shape[0] * input_tokens.shape[1]) # input_tokens.shape[0] * input_tokens.shape[1] = B*num_xs_xt*L (B*L repeated num_xs_xt times)
    return Loss(loss=loss,
                token_mask=attention_mask)

def get_loss_fn(*,
                config,
                mask_index,
                vocab_size,
                tokenizer,
                noise,
                backbone,
                adapter=None,
                tune_bb: bool=False,
                prob_path: Optional[MaskProbPath]=None) -> BaseLoss:
  """
  Returns a callable loss object based on the config.
  Usage:
      loss_fn = get_loss_fn(config=cfg, mask_index=..., ...)
      out = loss_fn(x0, attention_mask)
  """
  # pretrain mode (discrete diffusion cross-entropy)
  if config.mode == "pretrain":
    return PretrainCrossEntropyLoss(
      config=config, mask_index=mask_index, vocab_size=vocab_size,
      tokenizer=tokenizer, noise=noise, backbone=backbone
    )
  
  elif config.mode == 'evaluation':
    return 0

  # adapter fine-tuning mode
  lt = config.adapter_training.loss_type
  if lt == "token-critic":
    return AdapterFinetuneTokenCriticLoss(
      config=config, mask_index=mask_index, vocab_size=vocab_size,
      tokenizer=tokenizer, noise=noise, backbone=backbone,
      adapter=adapter, tune_bb=tune_bb, prob_path=prob_path
    )
  elif lt == "prism-loss":
    return AdapterFinetunePRISMLoss(
      config=config, mask_index=mask_index, vocab_size=vocab_size,
      tokenizer=tokenizer, noise=noise, backbone=backbone,
      adapter=adapter, tune_bb=tune_bb, prob_path=prob_path
    )
  else:
    raise ValueError(f"Unknown adapter_training.loss_type: {lt!r}")