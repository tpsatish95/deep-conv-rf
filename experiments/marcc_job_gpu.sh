#!/bin/bash
#SBATCH --job-name=DeepConvRFGPU
#SBATCH -N 1
#SBATCH -n 18
#SBATCH -p gpuk80
#SBATCH --gres=gpu:3
#SBATCH -t 7:0:0
#SBATCH --mail-type=end
#SBATCH --mail-user=spalani2@jhu.edu

module load cuda/9.0
module load singularity

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

singularity pull --name pytorch.simg shub://marcc-hpc/pytorch:0.4.1
singularity exec --nv ./pytorch.simg python -u run.py

# Notes:
# - all 4 GPU exps take 6 hours on the above config
# - sbatch marcc_job_gpu.sh
# - sqme
# - after state changes from PD to R
# - tail -f slurm-*.out
