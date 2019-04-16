#!/bin/bash
#SBATCH --job-name=DeepConvRFCPU
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --partition=parallel
#SBATCH -t 48:0:0
#SBATCH --mail-type=end
#SBATCH --mail-user=spalani2@jhu.edu

module load cuda/9.0
module load singularity

# redefine SINGULARITY_HOME to mount current working directory to base $HOME
export SINGULARITY_HOME=$PWD:/home/$USER

singularity pull --name pytorch.simg shub://marcc-hpc/pytorch:0.4.1
singularity exec --nv ./pytorch.simg python -u DeepConvSharedRFBaselineCIFAR10.py

# Notes:
# - sbatch marcc_job_cpu.sh
# - sqme
# - after state changes from PD to R
# - tail -f slurm-*.out
