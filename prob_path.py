import torch
import math
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from models.model_wrapper import DITWrapper, FeatureDITWrapper

def _sample_categorical(categorical_probs: Tensor) -> Tensor:
  r"""Sample from a categorical distribution using the Gumbel-max trick.
  Args:
    categorical_probs: A tensor of shape `(..., num_classes)` representing
      categorical probabilities.
  Returns:
    A tensor of shape `(...)` representing sampled class indices.
  """
  
  categorical_probs = categorical_probs.to(torch.float64)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


class MaskProbPath:
  """Base class for all samplers."""
  def __init__(self, config, mask_index, vocab_size, noise, backbone, adapter=None):
      self.config = config
      self.solver = config.sampling.predictor
      self.sampling_strategy = config.sampling.sampler
      self.mask_index = mask_index
      self.vocab_size = vocab_size
      self.noise = noise
      self.parameterization = self.config.parameterization
      self.time_conditioning = config.time_conditioning
      self.wrapped_backbone= DITWrapper(model = backbone,
                                        time_conditioning = self.time_conditioning,
                                        is_adapter = False)
      self.wrapped_adapter = None
      if adapter is not None:
        if self.config.adapter.input_type == 'xt' or self.config.adapter.input_type == 'x0':
          self.wrapped_adapter = DITWrapper(model = adapter,
                                            time_conditioning = self.time_conditioning,
                                            is_adapter = True)
        else: #embedding input
          self.wrapped_adapter = FeatureDITWrapper(model = adapter)
      self.neg_infinity = -1000000.0
      self.num_fixed_tokens = config.model.length - config.sampling.num_initial_masks
        
  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.long)
  
  def _p_x0_xt(self, logits, x):
    logits[:, :, self.mask_index] += self.neg_infinity
    probs = F.softmax(logits, dim=-1).detach()
    unmasked_indices = (x != self.mask_index)
    # xt -> x0_theta carry over   
    probs[unmasked_indices] = 0
    probs[unmasked_indices, x[unmasked_indices]] = 1
    return probs
  
  def _schedule_k(self, total_mask_tokens, total_steps, remasking_s_on, remasking_steps, current_step):   
    if current_step < remasking_s_on:
      return min(math.ceil((total_mask_tokens / total_steps) * (current_step + 1)+self.num_fixed_tokens), self.config.model.length)
    elif current_step >= remasking_s_on and current_step < (remasking_s_on + remasking_steps):
      return min(math.ceil((total_mask_tokens / total_steps) * remasking_s_on+self.num_fixed_tokens), self.config.model.length)
    else:
      return min(math.ceil((total_mask_tokens / total_steps) * (current_step + 1 - remasking_steps)+self.num_fixed_tokens), self.config.model.length)
  
  def _get_remasking_k(self, num_generated_tokens):
    if self.config.sampling.eta > 0.0:
      k = self.config.model.length / self.config.sampling.steps
      alpha_t = num_generated_tokens / self.config.model.length
      alpha_s = min((k + num_generated_tokens) / self.config.model.length, 1.0)
      if alpha_t > 0:
        sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
      else:
        sigma = self.config.sampling.eta
      binom_dist = torch.distributions.Binomial(total_count=num_generated_tokens, probs=sigma)
      num_remasking_tokens = binom_dist.sample().to(torch.int).item()
    else:
      num_remasking_tokens = self.config.sampling.num_remask
    return num_remasking_tokens

  def _compute_model_outputs(self, x, sigma, mode='sample'):
    '''
    forward function that returns the per-token mask posteriors, logits and confidences.
    '''
    conf = None
    logits, hidden_states, c = self.wrapped_backbone.get_output_and_features(x, sigma)
    probs = self._p_x0_xt(logits, x)

    if mode == 'sample' and self.wrapped_adapter is not None:
      if self.config.adapter.input_type == 'xt':
        conf = self.wrapped_adapter.get_score(x, sigma)
      elif self.config.adapter.input_type == 'embedding':
        conf = self.wrapped_adapter.get_score(hidden_states, c)
      else: #x0 input: tentative
        conf = torch.rand(*logits.shape[:-1], device=logits.device)
    else:
      # assign random confidence if no adapter is used
      conf = torch.rand(*logits.shape[:-1], device=logits.device)

    return probs, conf, logits    
    
  @torch.inference_mode()
  def generate_sample(self, num_steps=None, eps=1e-5, test_ds=None, device='cpu', return_intermediates=False):
    """Generate samples from the model."""
    batch_size = self.config.sampling.sample_batch_size
    
    if test_ds is not None:
      # Get conditioned batch from test dataset (Sudoku)
      test_batch = test_ds["input_ids"]
      x = test_batch[:batch_size].to(device)
    else:
      x = self._sample_prior(
        batch_size,
        self.config.model.length).to(device)
    fixed_tokens = (x != self.mask_index)
    
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=device)
    if num_steps !=0:
      dt = (1 - eps) / num_steps
    p_x0_cache = None
    min_t = timesteps[-1].item()
    x_list = []
    if return_intermediates:
      x_list.append(x.clone())
    if self.config.sampling.dfm:
      timestep = 1
      while timestep > dt:
        t = timestep * torch.ones(x.shape[0], 1, device=device)
        p_x0_cache, x_next, timestep = self._dfm_caching_update(x, t, p_x0=p_x0_cache)
        x = x_next
      min_t = timestep
    else:
      confident_score = - torch.ones_like(x, device=device).to(torch.bfloat16) * torch.inf
      num_clean_tokens = [self._schedule_k(self.config.sampling.num_initial_masks,
                                        self.config.sampling.steps,
                                        self.config.sampling.loop_step_on,
                                        self.config.sampling.loop_steps, i) for i in range(num_steps)]
      for i in tqdm(range(num_steps)):
        t = timesteps[i] * torch.ones(
          x.shape[0], 1, device=device)
        if self.solver == 'ddpm_cache':
          p_x0_cache, x_next, confident_score = self._ddpm_caching_update(
            x, t, dt, p_x0=p_x0_cache, conf=confident_score, fixed_tokens=fixed_tokens)
          x = x_next
        elif self.solver == 'topk_static':
          x_next, _ = self._topk_static_update(
            x, t, k=num_clean_tokens[i], step=i, fixed_tokens=fixed_tokens)
          x = x_next
        else:
          raise NotImplementedError(f'Sampler {self.solver} is not supported')
        if return_intermediates:
          x_list.append(x.clone())

    
    # If mask still exists, remove noise with argmax
    if self.config.sampling.noise_removal:
      t = min_t * torch.ones(x.shape[0], 1, device=device)
      unet_conditioning = self.noise(t)[0]
      x = self._compute_model_outputs(x, unet_conditioning)[0].argmax(dim=-1)
    
    return x, x_list
  
  def _score_tokens(self, p_x0, masks, conf=None, step=None, x0=None, mode='sample', fixed_tokens=None):
    '''
    score all tokens with the following rule and choose top-k tokens, rest of the tokens will be masked.
    fixed tokens (prompt or initial condition): torch.inf
    masked tokens: score \in [0,1]
    clean tokens that will be remasked: score \in [-1, 0]
    clean tokens that will remain clean: torch.inf
    '''
    if mode == 'adapter_finetune':
      # random scoring during adapter fine-tuning
      token_score = torch.rand(*p_x0.shape[:-1], device=p_x0.device)
      # greedy scoring during adapter fine-tuning
      # token_score = torch.gather(p_x0, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
      token_score[~masks] = torch.inf
      return token_score

    if step >= self.config.sampling.step_on and step < self.config.sampling.step_off:
      if self.sampling_strategy == 'topk_prob':
        token_score = torch.gather(p_x0, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        token_score[~masks] = torch.inf
      elif self.sampling_strategy == 'topk_prob_margin':
        top2_p_1t, _ = torch.topk(p_x0, k=2, dim=-1)
        token_score = top2_p_1t[..., 0] - top2_p_1t[..., 1]
        token_score[~masks] = torch.inf
      elif self.sampling_strategy == 'ar':
        token_score = torch.zeros(p_x0.shape[:-1], device=p_x0.device)
        token_score[..., step] = 1.0
        token_score[~masks] = torch.inf
      elif self.sampling_strategy == 'topk_hybrid_score':
        clean_token_confidence = conf
        mask_token_confidence = torch.gather(p_x0, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        token_score = torch.where(masks, mask_token_confidence, clean_token_confidence)
      elif self.sampling_strategy == 'token-critic':
        if self.config.adapter.input_type == 'x0': #token critic with x0 input
          sigma = torch.zeros(x0.shape[0], device=x0.device)
          token_score = self.wrapped_adapter.get_score(x0, sigma)
        else: #token critic with xt input
          token_score = conf       
      else: #random, prism
        token_score = torch.rand(*p_x0.shape[:-1], device=p_x0.device)
        token_score[~masks] = torch.inf
    else:
      token_score = torch.rand(*p_x0.shape[:-1], device=p_x0.device)
      token_score[~masks] = torch.inf
    
    #modify score of clean tokens for remasking
    if self.sampling_strategy == 'prism':
      if step >= self.config.sampling.step_on and step < self.config.sampling.step_off:  
        remask_filter = ~masks & ~fixed_tokens
        num_generated_tokens = remask_filter.sum(dim=-1)[0]
        num_remasking_tokens = self._get_remasking_k(num_generated_tokens.item())
        if num_remasking_tokens > 0 and \
        num_generated_tokens > num_remasking_tokens and \
        num_generated_tokens < self.config.sampling.num_initial_masks - num_remasking_tokens:
          remask_score = 1.0 - conf
          remask_score[~remask_filter] = -torch.inf
          _, remask_dim = torch.topk(remask_score, k=num_remasking_tokens, dim=-1, largest=True)
          selected_to_mask = torch.zeros_like(remask_score, dtype=torch.bool)
          selected_to_mask = selected_to_mask.scatter_(1, remask_dim, True)
          token_score = torch.where(selected_to_mask, -remask_score, token_score)
    token_score[fixed_tokens] = torch.inf
    return token_score

  def _select_k_tokens(self, p_x0, masks, conf=None, k=1, step=0, x0=None, fixed_tokens=None, mode='sample'):
    token_score = self._score_tokens(p_x0, masks, conf, step, x0, mode, fixed_tokens)    
    if torch.is_tensor(k) and k.shape[0] > 1: #adapter fine-tuning
      k_eff = k.to(torch.long)  # (B,)
      k_max = int(k_eff.max().item())
      _, change_dim = torch.topk(token_score, k=k_max, dim=-1, largest=True)
      keep = torch.arange(k_max, device=change_dim.device).unsqueeze(0) < k_eff.unsqueeze(1)
      selected_tokens = torch.zeros_like(token_score, dtype=torch.bool)
      selected_tokens.scatter_(1, change_dim, keep).to(torch.bool)
    else: #top-k static
      _, change_dim = torch.topk(token_score, k=k, dim=-1, largest=True)
      selected_tokens = torch.zeros_like(token_score, dtype=torch.bool)
      selected_tokens = selected_tokens.scatter_(1, change_dim, True).to(torch.bool)
    return selected_tokens

  def _topk_static_update(self, x, t, k=1, step=0, fixed_tokens=None, mode='sample'):
    '''
    update x to have k clean tokens (not masked tokens).
    '''
    assert self.time_conditioning == False
    sigma_t, _ = self.noise(t)
    masks = (x == self.mask_index).to(torch.bool)
    
    if fixed_tokens is None:
      fixed_tokens = torch.zeros_like(masks, dtype=torch.bool)
    
    # PRISM-loop stage (remask random k tokens)
    if mode=='sample' and (step >= self.config.sampling.loop_step_on
        and step < (self.config.sampling.loop_step_on + self.config.sampling.loop_steps)):
      remasking_score = torch.rand(*x.shape, device=x.device)
      remasking_score[masks] = -torch.inf
      remasking_score[fixed_tokens] = -torch.inf
      num_remasking_tokens = self.config.sampling.num_remask_loop
      _, selected_indices = torch.topk(remasking_score, k=num_remasking_tokens, dim=1)
      selected_to_mask = torch.zeros_like(masks, dtype=torch.bool)
      selected_to_mask.scatter_(1, selected_indices, True)
      x[selected_to_mask] = self.mask_index
      masks = (x == self.mask_index).to(torch.bool)

    p_x0, conf, logits = self._compute_model_outputs(x, sigma_t, mode=mode)
    p_x0_before_nucleus = p_x0.clone()
    
    if self.config.sampling.nucleus_p < 1:
      sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
      top_p_mask[..., 0] = True
      nucleus_probs = sorted_probs * top_p_mask
      p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)

    if mode == 'adapter_finetune':
      num_xs_xt = self.config.sampling.num_x0_xt * self.config.sampling.num_xs_x0
      p_x0 = p_x0.repeat(self.config.sampling.num_x0_xt, 1, 1)
      x0_theta = _sample_categorical(p_x0)
      x_next = x.clone().repeat(num_xs_xt, 1)
      p_x0_before_nucleus = p_x0_before_nucleus.repeat(num_xs_xt, 1, 1)
      masks = masks.repeat(num_xs_xt, 1)
      conf = conf.repeat(num_xs_xt, 1)
      x0_theta = x0_theta.repeat(self.config.sampling.num_xs_x0, 1)
      fixed_tokens = fixed_tokens.repeat(num_xs_xt, 1)
      k = k.repeat(num_xs_xt)
    else:
      x0_theta = _sample_categorical(p_x0)
      x_next = x.clone()

    clean_token_filter = self._select_k_tokens(p_x0_before_nucleus, masks, conf, k, step, x0_theta, fixed_tokens=fixed_tokens, mode=mode)
    x_next = torch.where(clean_token_filter, x0_theta, self.mask_index)
    return x_next, logits

  def _ddpm_caching_update(self, x, t, dt, p_x0=None, conf=None, fixed_tokens=None):
    '''
    This function is used to update the model's output for the next step, therefore this function
    should be called n_step times.
    '''
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0, _, _ = self._compute_model_outputs(x, sigma_t)
      if self.config.sampling.nucleus_p < 1:
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    assert move_chance_t.ndim == p_x0.ndim

    if self.sampling_strategy == 'mdlm':
      q_xs = p_x0 * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      _x = _sample_categorical(q_xs)
      copy_flag = (x != self.mask_index).to(x.dtype)
      xs = copy_flag * x + (1 - copy_flag) * _x
    elif self.sampling_strategy == 'forward-backward':
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma = (alpha_s - alpha_t) / alpha_t
      else:
        sigma = 1
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.sampling_strategy == 'remdm-cap':
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma = min(self.config.sampling.eta, (1 - alpha_s) / alpha_t)
      else:
        sigma = self.config.sampling.eta
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.sampling_strategy == 'remdm-rescale':
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      sigma = self.config.sampling.eta * sigma_max
      q_xs = p_x0 * (1 - sigma)
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
    elif self.sampling_strategy == 'remdm-conf':
      alpha_t = (1 - move_chance_t)[0].item()
      alpha_s = (1 - move_chance_s)[0].item()
      if alpha_t > 0:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
      else:
        sigma_max = 1
      temp=1.0
      eta = (conf/temp).softmax(dim=-1)
      masked_flag = (x == self.mask_index).to(torch.bool)
      eta[masked_flag] = 0
      sigma = eta * sigma_max
      q_xs = p_x0 * (1 - sigma[:, :, None])
      q_xs[..., self.mask_index] = sigma
      q_xs_2 = p_x0 * ((alpha_s - (1 - sigma[:, :, None]) * alpha_t) / (1 - alpha_t))
      q_xs_2[..., self.mask_index] = (1 - alpha_s - sigma * alpha_t) / (1 - alpha_t)
      copy_flag = (x != self.mask_index).to(torch.bool)
      q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
      xs = _sample_categorical(q_xs)
      # update conf
      unmask_mask = (x == self.mask_index) & (xs != self.mask_index)
      batch_indices = torch.arange(xs.shape[0])[:, None]
      feature_indices = torch.arange(xs.shape[1])
      conf_values = - p_x0[batch_indices, feature_indices, xs]
      conf[unmask_mask] = conf_values[unmask_mask]
      remask_mask = (x != self.mask_index) & (xs == self.mask_index)
      conf[remask_mask] = -torch.inf
    elif self.sampling_strategy == 'remdm-loop':
      time = t[0].item()
      # compute alpha_t and alpha_s
      if time > self.config.sampling.t_on:
        move_chance_t = (1 - (1 - t) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
        move_chance_s = (1 - (1 - t + dt) * self.config.sampling.alpha_on / (1 - self.config.sampling.t_on))[:, None, None]
      elif time <= self.config.sampling.t_off:
        move_chance_t = (t * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
        move_chance_s = ((t - dt) * (1 - self.config.sampling.alpha_on) / self.config.sampling.t_off)[:, None, None]
      else:
        move_chance_t, move_chance_s = None, None
      if time > self.config.sampling.t_on or time <= self.config.sampling.t_off: # MDLM scheduling
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        xs = copy_flag * x + (1 - copy_flag) * _x
      else: # ReMDM scheduling
        sigma = self.config.sampling.eta
        q_xs = p_x0 * (1 - sigma)
        q_xs[..., self.mask_index] = sigma
        q_xs_2 = p_x0 * ((self.config.sampling.alpha_on - (1 - sigma) * self.config.sampling.alpha_on) / (1 - self.config.sampling.alpha_on))
        q_xs_2[..., self.mask_index] = (1 - self.config.sampling.alpha_on - self.config.sampling.alpha_on * sigma) / (1 - self.config.sampling.alpha_on)
        copy_flag = (x != self.mask_index).to(torch.bool)
        q_xs = torch.where(copy_flag.unsqueeze(-1), q_xs, q_xs_2)
        xs = _sample_categorical(q_xs)
    xs = torch.where(fixed_tokens, x, xs)
      
    if torch.allclose(xs, x) and not self.time_conditioning:
      p_x0_cache = p_x0
    else:
      p_x0_cache = None
    return p_x0_cache, xs, conf

  def _dfm_caching_update(self, x, t, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    timestep = t[0].item()
    # Hard-coded corrector schecule, see Appendix D in https://proceedings.neurips.cc/paper_files/paper/2024/file/f0d629a734b56a642701bba7bc8bb3ed-Paper-Conference.pdf
    at = 10 * timestep ** 0.25 * (1 - timestep) ** 0.25 + 1
    bt = at - 1
    if timestep == 1:
      dt = 1 / self.config.sampling.steps
    else:
      dt = min(1 / self.config.sampling.steps, 1 / (at / timestep  + bt / (1 - timestep)))
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3
    if p_x0 is None:
      p_x0, _, _ = self._compute_model_outputs(x, sigma_t)
      if self.config.sampling.nucleus_p < 1:
        sorted_probs, sorted_indices = torch.sort(p_x0, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= self.config.sampling.nucleus_p
        top_p_mask[..., 0] = True
        nucleus_probs = sorted_probs * top_p_mask
        nucleus_probs /= nucleus_probs.sum(dim=-1, keepdim=True)
        p_x0 = torch.zeros_like(p_x0).scatter_(-1, sorted_indices, nucleus_probs)
    
    assert move_chance_t.ndim == p_x0.ndim

    if timestep == 1:
      q_xs = p_x0 * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      _x = _sample_categorical(q_xs)
    
      copy_flag = (x != self.mask_index).to(x.dtype)
      xs = copy_flag * x + (1 - copy_flag) * _x
      if torch.allclose(xs, x) and not self.time_conditioning:
        p_x0_cache = p_x0
      else:
        p_x0_cache = None
      return p_x0_cache, xs, timestep - dt

    alpha_t = 1 - move_chance_t[0].item()
    alpha_s = 1 - move_chance_s[0].item()
    coef1 = at * (alpha_s - alpha_t) / (1 - alpha_t)
    q_xs = p_x0 * coef1
    q_xs[..., self.mask_index] = 1 - coef1
    coef2 = 1 - bt * (alpha_s / alpha_t - 1)
    q_xs2 = p_x0 * coef2
    q_xs2[..., self.mask_index] = 1 - coef2
    masked_flag = (x == self.mask_index).to(torch.bool)
    q_xs = torch.where(masked_flag.unsqueeze(-1), q_xs, q_xs2)
    xs = _sample_categorical(q_xs)
  
    if torch.allclose(xs, x) and not self.time_conditioning:
      p_x0_cache = p_x0
    else:
      p_x0_cache = None

    return p_x0_cache, xs, timestep - dt
