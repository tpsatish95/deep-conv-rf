#!/bin/bash
#SBATCH --job-name=DeepConvRFGPU
#SBATCH -N 1
#SBATCH -n 18
#SBATCH -p gpuk80
#SBATCH --gres=gpu:3
#SBATCH -t 20:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=spalani2@jhu.edu

module load cuda/9.0
module load singularity

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

singularity pull --name pytorch.simg shub://marcc-hpc/pytorch:0.4.1
singularity exec --nv ./pytorch.simg python -u run.py FashionMNIST 7 9

# Notes:
# - sbatch marcc_job_gpu.sh
# - sqme
# - after state changes from PD to R
# - tail -f slurm-*.out
