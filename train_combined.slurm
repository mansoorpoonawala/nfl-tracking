#!/bin/bash
#SBATCH --job-name=nfl_yolo11_combined
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --qos=regular
#SBATCH --account=m4431_g
#SBATCH --constraint=gpu
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

module load python
source venv/bin/activate

mkdir -p logs

cd /pscratch/sd/m/mpoona/nfl_training
python train_combined.py
