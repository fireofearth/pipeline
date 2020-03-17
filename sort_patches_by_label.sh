#!/bin/bash
#SBATCH --job-name sort_patches_by_label
#SBATCH --cpus-per-task 1
#SBATCH --output /home/cochen/cchen/ml/slurm/%j.out
#SBATCH --error /home/cochen/cchen/ml/slurm/%j.err
#SBATCH -w dlhost02
#SBATCH -p dgxV100
#SBATCH --gres=gpu:1
#SBATCH --time=4-90:00:00
#SBATCH --chdir /home/cochen/cchen/ml/pipeline

source /home/cochen/cchen/py2

kronos run \
	-c $PWD/../ \
	-y sort_patches_by_label.yaml \
	--no_prefix 