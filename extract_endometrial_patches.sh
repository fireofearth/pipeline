#!/bin/bash
#SBATCH --job-name extract_endometrial_patches
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
	-y extract_endometrial_patches.yaml \
	--no_prefix 
