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
# Set visible GPUs to 0,1,2,4
export CUDA_VISIBLE_DEVICES=0
YOUR_BASE_PATH="/home/local/eda14/sk58348/projects/diffusion/discrete/PRISM"
checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/sudoku_mdlm_100k.ckpt

T=0
sampling_steps=51
p=1.0

python -u -m main \
  seed=0 \
  mode=evaluation \
  loader.global_batch_size=256 \
  loader.eval_global_batch_size=1000 \
  loader.batch_size=256 \
  loader.eval_batch_size=1000 \
  model=tiny \
  data=sudoku \
  data.cache_dir="/home/local/eda14/sk58348/.cache/huggingface/datasets" \
  wandb.name=sudoku_sample_remdm-conf \
  parameterization=subs \
  model.length=89 \
  eval.generate_samples=True \
  eval.compute_success_rate=True \
  eval.checkpoint_path=${checkpoint_path} \
  T=${T} \
  sampling.steps=${sampling_steps} \
  sampling.num_sample_batches=1 \
  sampling.nucleus_p=${p} \
  sampling.predictor="ddpm_cache" \
  sampling.noise_removal=true \
  sampling.sampler="remdm-conf" \
  sampling.sample_batch_size=1000 \
  sampling.num_initial_masks=51 \
  adapter.enable=false \