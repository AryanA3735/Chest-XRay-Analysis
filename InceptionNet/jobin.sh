#! /usr/bin/env bash
#SBATCH -N 1
#SBATCH --time=4-00:00
#SBATCH --job-name=codiff2
#SBATCH --ntasks-per-node=16
#SBATCH --error=error.txt
#SBATCH --output=outin.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load DL-Conda_3.7
cd $SLURM_SUBMIT_DIR

nvidia-smi > nv.txt

source /home/apps/DL/DL-CondaPy3.7/bin/activate myenv

# CUDA_VISIBLE_DEVICES=1 python main.py --logdir 'outputs/512_codiff_mask_text' --base 'configs/512_codiff_mask_text.ya$

# CUDA_VISIBLE_DEVICES=1 python generate_512.py --mask_path test_data/512_masks/27007.png --input_text "This man has be$

CUDA_VISIBLE_DEVICES=0,1 python train.py &> outin.txt
