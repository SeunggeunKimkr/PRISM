import itertools
import hydra.utils
import lightning as L
import torch
import transformers
import copy

import dataloader
import models
import noise_schedule
from eval_metric import build_metric_collections, Perplexity, SudokuEvaluator, TextEvaluator
from prob_path import MaskProbPath
from loss import get_loss_fn
import mauve
import json

class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.data = self.config.data.train
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    self.T = self.config.T
    self.subs_masking = self.config.subs_masking
    
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.adapter = None
    self.tune_bb = getattr(self.config.adapter_training, "tune_backbone", False)
    if self.config.adapter.enable:
      if self.config.adapter.input_type == 'xt' or self.config.adapter.input_type == 'x0':
        if self.config.adapter.backbone == 'tiny_dit':
          self.adapter_config = copy.deepcopy(self.config)
          self.adapter_config.model.hidden_size = 256
          self.adapter_config.model.n_heads = 4
          self.adapter_config.model.n_blocks = 4
          self.adapter = models.dit.DIT(self.adapter_config, vocab_size=self.vocab_size) 
          self.adapter.output_layer = models.dit.DDitFinalLayer(
            self.adapter_config.model.hidden_size,
            1,
            self.adapter_config.model.cond_dim)
        else:
          raise ValueError(f'Raw input adapter only supports DIT backbone.')
      elif self.config.adapter.input_type == 'embedding':
        if self.config.adapter.backbone == 'adapter_dit':
          self.adapter_config = copy.deepcopy(self.config)
          self.adapter_config.model.n_blocks = self.config.adapter.n_blocks
          self.adapter = models.dit.FeatureDIT(self.adapter_config, output_size=1)
        else:
          raise ValueError(f'Unknown adapter type: {self.config.adapter.backbone}')
      else:
        raise ValueError(f'Unknown adapter input type: {self.config.adapter.input_type}')
      self.adapter.train()
      if not self.tune_bb:
        for param in self.backbone.parameters():
          param.requires_grad = False
            
    # metrics are automatically reset at end of epoch
    self.train_metrics, self.valid_metrics, self.test_metrics = build_metric_collections()

    if self.data == 'sudoku':
      self.evaluator = SudokuEvaluator()
    else:
      self.evaluator = TextEvaluator(self.config)
      self.gen_ppl_metric = Perplexity()

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      if self.config.adapter.enable:
        params_to_ema = [self.adapter.parameters(), self.noise.parameters()]
        if self.tune_bb:
          params_to_ema.append(self.backbone.parameters())
        self.ema = models.ema.ExponentialMovingAverage(
          itertools.chain(*params_to_ema),
            decay=self.config.training.ema)
      else:
        self.ema = models.ema.ExponentialMovingAverage(
            itertools.chain(self.backbone.parameters(),
                            self.noise.parameters()),
            decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.adapter_sampling_eps = self.config.adapter_training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self.ProbPath = MaskProbPath(self.config,
                                self.mask_index,
                                self.vocab_size,
                                self.noise,
                                self.backbone,
                                self.adapter)
    self.loss_fn = get_loss_fn(config=self.config,
                                mask_index=self.mask_index,
                                vocab_size=self.vocab_size,
                                tokenizer=self.tokenizer,
                                noise=self.noise,
                                backbone=self.backbone,
                                adapter=self.adapter,
                                tune_bb=self.tune_bb,
                                prob_path=self.ProbPath)
    self._validate_configuration()

  def _load_adapter_model(self, checkpoint_path):
    print(f"Loading adapter model from {checkpoint_path}")
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if 'ema' in checkpoint and self.ema:
      # Load EMA state dict if available
      self.ema.load_state_dict(checkpoint['ema'])
      
    # Load weights for adapter model
    adapter_state_dict = {}
    for key, value in state_dict.items():
      if key.startswith('adapter.'):
        new_key = key.replace('adapter.', '')
        adapter_state_dict[new_key] = value
      elif key.startswith('noise.'):
        new_key = key.replace('noise.', '')
        adapter_state_dict[new_key] = value
    
    # Load weights
    missing_keys, unexpected_keys = self.adapter.load_state_dict(
      adapter_state_dict, strict=False)
    if missing_keys:
      print(f"Warning: Missing keys in adapter model: {missing_keys}")
    if unexpected_keys:
      print(f"Warning: Unexpected keys in adapter model: {unexpected_keys}")

    self.adapter.eval()
    print("Adapter model loaded successfully")
    
  def load_adapter_model(self, checkpoint_path):
    """Load adapter model from checkpoint after initialization."""
    self._load_adapter_model(checkpoint_path)
    

  ############################################################
  # Validate Configuration
  ############################################################
  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization in {'sedd', 'd3pm', 'ar'}:
      raise NotImplementedError(self.parameterization + ' training is not supported')
      assert not self.importance_sampling
      assert not self.change_of_variables
      assert self.T > 0
    assert not self.change_of_variables
    assert not self.importance_sampling
    if self.T > 0:
      raise NotImplementedError('discrete time training is not supported')
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'
    if self.config.sampling.sampler == 'ar':
      assert self.config.sampling.steps == self.config.sampling.num_initial_masks
    if self.config.adapter.enable and self.config.adapter_training.loss_type == 'prism-loss':
      if not self.config.adapter_training.tune_backbone:
        assert self.config.adapter_training.reg_lambda == 0.0
    if self.config.sampling.sampler == 'prism':
      if self.config.sampling.eta > 0.0:
        assert self.config.sampling.num_remask == 0
      if self.config.sampling.num_remask > 0:
        assert self.config.sampling.eta == 0.0
    if self.config.adapter.enable:
      if self.config.adapter_training.loss_type == 'token-critic':
        assert self.config.adapter.input_type == 'x0'
      if self.config.adapter_training.loss_type == 'prism-loss':
        assert (self.config.adapter.input_type == 'xt' or self.config.adapter.input_type == 'embedding')

  def configure_optimizers(self):
    # Build param groups
    pg = []
    base_lr = self.config.optim.lr
    adapter_lr = getattr(self.config.optim, "adapter_lr", base_lr)
    backbone_lr = getattr(self.config.optim, "backbone_lr", base_lr)
    noise_lr = getattr(self.config.optim, "noise_lr", base_lr)

    if self.config.adapter.enable:
      if self.tune_bb:
        pg.append({"params": self.backbone.parameters(), "lr": backbone_lr})
      pg.append({"params": self.adapter.parameters(), "lr": adapter_lr})
      pg.append({"params": self.noise.parameters(), "lr": noise_lr})
    else:
      # Plain backbone training
      pg.append({"params": self.backbone.parameters(), "lr": backbone_lr})
      pg.append({"params": self.noise.parameters(), "lr": noise_lr})

    optimizer = torch.optim.AdamW(
        pg,
        lr=base_lr,  # global default; per-group 'lr' overrides it
        betas=(self.config.optim.beta1, self.config.optim.beta2),
        eps=self.config.optim.eps,
        weight_decay=self.config.optim.weight_decay,
    )

    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
        "scheduler": scheduler,
        "interval": "step",
        "monitor": "val/loss",
        "name": "trainer/lr",
    }
    return [optimizer], [scheduler_dict]


  ############################################################
  # Checkpoint (for backbone training)
  ############################################################
  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      if (not (self.config.adapter.enable)) or self.tune_bb: #TODO: why do we need this? for final layer only training (tune_bb=false)
        self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  ############################################################
  # Training Wrappers
  ############################################################
  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      if self.config.adapter.enable:
        if self.tune_bb:
          params_to_ema = [self.backbone.parameters(), self.adapter.parameters(), self.noise.parameters()]
        else:
          params_to_ema = [self.adapter.parameters(), self.noise.parameters()]
        self.ema.update(itertools.chain(*params_to_ema))
      else:
        self.ema.update(itertools.chain(
          self.backbone.parameters(),
          self.noise.parameters()))
      
  def on_train_epoch_start(self):
    if self.config.adapter.enable:
      self.adapter.train()
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss
  
  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = torch.ones_like(batch['input_ids'])
    
    losses = self.loss_fn(batch['input_ids'], attention_mask)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  
  ############################################################
  # Validation during training
  ############################################################
  def on_validation_epoch_start(self):
    if self.ema:
      if self.config.adapter.enable:
        if self.tune_bb:
          params_to_ema = [self.backbone.parameters(), self.adapter.parameters(), self.noise.parameters()]
        else:
          params_to_ema = [self.adapter.parameters(), self.noise.parameters()]
        self.ema.store(itertools.chain(*params_to_ema))
        self.ema.copy_to(itertools.chain(*params_to_ema))
      else:    
        self.ema.store(itertools.chain(
          self.backbone.parameters(),
          self.noise.parameters()))
        self.ema.copy_to(itertools.chain(
          self.backbone.parameters(),
          self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    if self.config.adapter.enable:
      self.adapter.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    if self.config.mode == 'pretrain':
      val_loss = self._compute_loss(batch, prefix='val')
      self.log(name='val/loss',
               value=val_loss.item(),
               on_step=False,
               on_epoch=True,
               sync_dist=True)
      return val_loss
    else:
      return torch.tensor(0.0)


  def on_validation_epoch_end(self):
    if ((self.config.eval.generate_sample_on_sanity
        or not self.trainer.sanity_checking) 
        and self.data == 'sudoku'
        and self.config.eval.generate_samples):
      test_loader = dataloader.get_dataloaders(
        self.config, self.tokenizer, skip_train=True, skip_valid=True, skip_test=False)[2]
      test_ds = iter(test_loader)
      # for _ in range(self.config.sampling.num_sample_batches):
      test_iter = next(test_ds)
      num_steps = self.config.sampling.steps + self.config.sampling.loop_steps
      samples, _ = self.ProbPath.generate_sample(num_steps=num_steps, test_ds=test_iter, device=self.device)
      success = self.evaluator.eval_sample(samples)
      success_rate = sum(success) / len(success)
      print(f'Sudoku Success Rate: {success_rate}')
      self.log('val/sudoku_sr',
                success_rate,
                on_epoch=True,
                on_step=False,
                reduce_fx='sum',
                sync_dist=True)
        
    if ((self.config.eval.generate_sample_on_sanity
        or not self.trainer.sanity_checking)
        and self.data == 'openwebtext-train'
        and self.config.eval.generate_samples):
      entropies = []
      local_text_samples = []
      for i in range(self.config.sampling.num_sample_batches):
        num_steps = self.config.sampling.steps + self.config.sampling.loop_steps
        samples, _ = self.ProbPath.generate_sample(num_steps=num_steps, device=self.device)
        text_samples = self.tokenizer.batch_decode(samples)
        local_text_samples.extend(text_samples)
        if self.config.eval.compute_generative_perplexity:
          self.evaluator.compute_generative_perplexity(text_samples, self.gen_ppl_metric, device=self.device)
        if self.config.eval.compute_entropy:
          entropy = self.evaluator.compute_entropy(samples)
          entropies.append(entropy)
        
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])      
      
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                  self.gen_ppl_metric,
                  on_epoch=True,
                  on_step=False,
                  sync_dist=True)
      
      if self.config.eval.compute_entropy:
        self.log('val/entropy',
                  sum(entropies) / len(entropies),
                  on_epoch=True,
                  on_step=False,
                  sync_dist=True)
      
      #mauve computation after gathering all text samples from all gpus  
      if self.config.eval.compute_mauve:
        if self.config.trainer.devices > 1:
          gathered_samples = [None for _ in range(self.config.trainer.devices)]
          torch.distributed.all_gather_object(gathered_samples, local_text_samples)
          global_text_samples = []
          for samples in gathered_samples:
            global_text_samples.extend(samples)
        else:
          global_text_samples = local_text_samples
        mauve_score = self.evaluator.compute_mauve(global_text_samples, tokenizer=self.tokenizer)
        mauve_score = mauve_score if mauve_score is not None else -1.0
        self.log('val/mauve',
                  mauve_score,
                  on_epoch=True,
                  on_step=False,
                  reduce_fx='max',
                  sync_dist=True)
    if self.ema:
      if self.config.adapter.enable:
        if self.tune_bb:
          params_to_ema = [self.backbone.parameters(), self.adapter.parameters(), self.noise.parameters()]
        else:
          params_to_ema = [self.adapter.parameters(), self.noise.parameters()]
        self.ema.restore(itertools.chain(*params_to_ema))
      else:
        self.ema.restore(
          itertools.chain(self.backbone.parameters(),
                          self.noise.parameters()))