#!/bin/bash
#SBATCH --nodes=1
#SBATCH --qos=interactive
#SBATCH --time=04:00:00
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --account=m4431_g
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

# Run training for 300 epochs
time torchrun --standalone --nproc_per_node=4 train.py \
  --out_dir=out-300epochs \
  --max_iters=300 \
  > training_log_300epochs.txt 2> training_log_300epochs.err

echo "Training complete!"
