#!/bin/bash
#SBATCH --job-name train_endometrial_tumornormal
#SBATCH --cpus-per-task 1
#SBATCH --output /home/cochen/cchen/ml/slurm/%j.out
#SBATCH --error /home/cochen/cchen/ml/slurm/%j.err
#SBATCH -w dlhost02
#SBATCH -p dgxV100
#SBATCH --gres=gpu:1
#SBATCH --time=4-90:00:00
#SBATCH --chdir /projects/ovcare/classification/cchen/ml/pipeline

source /projects/ovcare/classification/cchen/py2

kronos run \
	-c $PWD/../ \
	-i $PWD/input.txt \
	-s $PWD/setup.txt \
	-y train_endometrial_tumornormal.yaml \
	--no_prefix 