#!/bin/bash
#SBATCH --job-name=im_llada
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaeyeon_kim@g.harvard.edu
#SBATCH --output=/n/netscratch/sham_lab/Everyone/jay_llada/slurm_logs/%j.out
#SBATCH --error=/n/netscratch/sham_lab/Everyone/jay_llada/slurm_logs/%j.err

source ~/.bashrc
conda deactivate
conda activate jay_vlmdm
module load cuda/12.4.1-fasrc01

export NCCL_SOCKET_FAMILY=AF_INET
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=$SLURM_NODEID

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --ntasks-per-node=1 --gpus-per-task=4 \
  python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train.py \
    --wandb \
    --job_name="llada_PRISM"

