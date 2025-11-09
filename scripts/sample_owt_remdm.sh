#!/bin/bash
#SBATCH -J remdm-loop                 # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[3090|a5000|a6000|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export CUDA_VISIBLE_DEVICES=0,1,2,3

YOUR_BASE_PATH="/home/local/eda14/sk58348/projects/diffusion/discrete/PRISM"

# Set cache directories to user's home space
export HF_HOME="/home/local/eda14/sk58348/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/local/eda14/sk58348/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/home/local/eda14/sk58348/.cache/huggingface/datasets"

# Create cache directories if they don't exist
# mkdir -p "${HF_HOME}"
# mkdir -p "${TRANSFORMERS_CACHE}"
# mkdir -p "${HF_DATASETS_CACHE}"

export HYDRA_FULL_ERROR=1

checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/mdlm.ckpt

T=0
sampling_steps=256
p=0.9
eta=0.04
t_on=0.55
t_off=0.05
alpha_on=0.9
generated_seqs_path=${YOUR_BASE_PATH}/outputs/owt_mdlm_T-${sampling_steps}_topp-${p}.json

python -u -m main \
    seed=0 \
    mode=evaluation \
    loader.eval_batch_size=64 \
    model=small \
    data=openwebtext-split \
    data.cache_dir="/home/local/eda14/sk58348/.cache/huggingface/datasets" \
    wandb.name=owt-sample-64-remdm-cap \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    eval.perplexity_batch_size=64 \
    eval.compute_generative_perplexity=True \
    eval.compute_entropy=True \
    eval.compute_mauve=True \
    time_conditioning=false \
    hydra.run.dir="${PWD}/outputs/remdm-cap" \
    T=${T} \
    sampling.steps=${sampling_steps} \
    sampling.eta=${eta} \
    sampling.num_sample_batches=16 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.predictor="ddpm_cache" \
    sampling.noise_removal=true \
    sampling.sampler="remdm-cap" \
    sampling.t_on=${t_on} \
    sampling.t_off=${t_off} \
    sampling.alpha_on=${alpha_on} \
    sampling.sample_batch_size=1 \
    sampling.num_initial_masks=1024 \
    adapter.enable=false \