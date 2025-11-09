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

checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/prism_mdlm.ckpt
adapter_checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/prism_mdlm.ckpt

sampling_steps=256
step_on=128
step_off=256
eta=0.01
num_remask=0
p=0.9
loop_steps=0
generated_seqs_path=${YOUR_BASE_PATH}/outputs/owt_mdlm_T-${sampling_steps}_topp-${p}.json
python -u -m main \
    seed=0 \
    mode=evaluation \
    loader.eval_batch_size=64 \
    model=small \
    data=openwebtext-split \
    data.cache_dir="/home/local/eda14/sk58348/.cache/huggingface/datasets" \
    wandb.name=owt-sample-64-256steps-prism \
    parameterization=subs \
    backbone=dit \
    model.length=1024 \
    eval.checkpoint_path=${checkpoint_path} \
    eval.adapter_checkpoint_path=${adapter_checkpoint_path} \
    eval.perplexity_batch_size=64 \
    eval.compute_generative_perplexity=True \
    eval.compute_entropy=True \
    eval.compute_mauve=True \
    time_conditioning=false \
    hydra.run.dir="${PWD}/outputs/prism" \
    sampling.steps=${sampling_steps} \
    sampling.step_on=${step_on} \
    sampling.step_off=${step_off} \
    sampling.loop_steps=${loop_steps} \
    sampling.loop_step_on=${sampling_steps} \
    sampling.num_remask_loop=0 \
    sampling.eta=${eta} \
    sampling.num_remask=${num_remask} \
    sampling.num_sample_batches=16 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.predictor="topk_static" \
    sampling.noise_removal=false \
    sampling.sampler="prism" \
    sampling.sample_batch_size=1 \
    sampling.num_initial_masks=1024 \
    adapter.backbone=adapter_dit \
    adapter.input_type=embedding \
    adapter.n_blocks=1 \
    adapter.enable=true \