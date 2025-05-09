#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=2
#SBATCH --error=logs/pred_dv.err
#SBATCH --output=logs/pred_dv.out
#SBATCH --exclude=''
#SBATCH --gpus-per-node=1
#SBATCH --job-name=pred_dv
#SBATCH --mem-per-gpu=100000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --partition=gpuq
#SBATCH --time=1440


# Load Micromamba
source /gpfs/data/karczmar-lab/workspaces/rachelgordon/micromamba/etc/profile.d/micromamba.sh

# Activate your Micromamba environment
micromamba activate 3dseg

python predict.py --target-tissue dv --image /ess/scratch/scratch1/rachelgordon/3dseg_preprocessed_resized/ --input-mask /ess/scratch/scratch1/rachelgordon/3dseg_masks_resized/ --save-masks-dir /ess/scratch/scratch1/rachelgordon/3dseg_dv_masks/ --model-save-path trained_models/dv_model.pth