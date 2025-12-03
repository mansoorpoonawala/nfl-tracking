#!/bin/bash
#SBATCH --nodes=2
#SBATCH --qos=interactive
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=m4431_g
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

cd /pscratch/sd/m/mpoona/nfl_training

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export LOCAL_RANK=$SLURM_LOCALID

srun python train_combined_8gpu.py \
  > training_log_all_500ep_8gpu.txt 2> training_log_all_500ep_8gpu.err
