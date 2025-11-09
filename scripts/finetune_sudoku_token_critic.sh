#!/bin/bash
#SBATCH -J train_mdlm                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.
export CUDA_VISIBLE_DEVICES=1
YOUR_BASE_PATH="/home/local/eda14/sk58348/projects/diffusion/discrete/PRISM"
checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/sudoku_mdlm_100k.ckpt
python -u -m main \
  seed=0 \
  mode=adapter_finetune \
  data=sudoku \
  parameterization=subs \
  model=tiny \
  time_conditioning=false \
  callbacks.checkpoint_monitor.monitor="val/sudoku_sr" \
  callbacks.checkpoint_monitor.mode=max \
  data.cache_dir="/home/local/eda14/sk58348/.cache/huggingface/datasets" \
  eval.generate_samples=true \
  eval.generate_sample_on_sanity=true \
  eval.compute_success_rate=true \
  eval.checkpoint_path=${checkpoint_path} \
  loader.global_batch_size=256 \
  loader.eval_global_batch_size=1000 \
  loader.batch_size=256 \
  loader.eval_batch_size=1000 \
  model.length=89 \
  wandb.name=sudoku_finetune_token-critic \
  sampling.steps=51 \
  sampling.num_sample_batches=1 \
  sampling.step_on=0 \
  sampling.step_off=51 \
  sampling.nucleus_p=1.0 \
  sampling.predictor="topk_static" \
  sampling.noise_removal=false \
  sampling.sampler="token-critic" \
  sampling.sample_batch_size=1000 \
  sampling.num_initial_masks=51 \
  sampling.num_x0_xt=1 \
  sampling.num_xs_x0=1 \
  trainer.val_check_interval=1.0 \
  trainer.max_steps=20_000 \
  optim.backbone_lr=3e-4 \
  optim.adapter_lr=3e-4 \
  adapter.enable=true \
  adapter.backbone="tiny_dit" \
  adapter.input_type="x0" \
  adapter_training.loss_type="token-critic" \
  adapter_training.num_demasking_tokens_per_step=89 \
  adapter_training.tune_backbone=false \
  adapter_training.reg_lambda=0.0 \