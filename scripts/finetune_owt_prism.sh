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
YOUR_BASE_PATH="/home/local/eda14/sk58348/projects/diffusion/discrete/PRISM"
checkpoint_path=${YOUR_BASE_PATH}/outputs/checkpoints/mdlm.ckpt
export CUDA_VISIBLE_DEVICES=0,1,2,3

sampling_steps=256
p=0.9
generated_seqs_path=${YOUR_BASE_PATH}/outputs/owt_ft_mdlm_T-${sampling_steps}_topp-${p}.json
  python -u -m main \
    backbone=dit \
    seed=0 \
    mode=adapter_finetune \
    data=openwebtext-split \
    parameterization=subs \
    model=small \
    time_conditioning=false \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    callbacks.checkpoint_monitor.monitor="val/mauve" \
    callbacks.checkpoint_monitor.mode=max \
    data.cache_dir="/home/local/eda14/sk58348/.cache/huggingface/datasets" \
    eval.generate_samples=True \
    eval.generate_sample_on_sanity=True \
    eval.compute_generative_perplexity=True \
    eval.compute_entropy=True \
    eval.compute_mauve=True \
    eval.checkpoint_path=${checkpoint_path} \
    eval.perplexity_batch_size=64 \
    loader.global_batch_size=256 \
    loader.eval_global_batch_size=256 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    model.length=1024 \
    wandb.name=owt-finetune-prism-finalchk \
    sampling.steps=${sampling_steps} \
    sampling.num_sample_batches=1 \
    sampling.step_on=128 \
    sampling.step_off=256 \
    sampling.loop_steps=0 \
    sampling.loop_step_on=256 \
    sampling.num_remask=0 \
    sampling.num_remask_loop=0 \
    sampling.eta=0.01 \
    sampling.generated_seqs_path=${generated_seqs_path} \
    sampling.nucleus_p=${p} \
    sampling.predictor="topk_static" \
    sampling.noise_removal=false \
    sampling.sampler="prism" \
    sampling.sample_batch_size=16 \
    sampling.num_initial_masks=1024 \
    sampling.num_x0_xt=1 \
    sampling.num_xs_x0=4 \
    trainer.val_check_interval=4000 \
    trainer.max_steps=20_000 \
    optim.backbone_lr=1.5e-4 \
    optim.adapter_lr=1.5e-4 \
    adapter.backbone=adapter_dit \
    adapter.n_blocks=1 \
    adapter.input_type=embedding \
    adapter.enable=true \
    adapter_training.loss_type=prism-loss \
    adapter_training.num_demasking_tokens_per_step=8 \
    adapter_training.tune_backbone=true \
    adapter_training.reg_lambda=0.5 \