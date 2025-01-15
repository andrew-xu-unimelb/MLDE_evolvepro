#!/bin/bash

#SBATCH --job-name=embedding_trial_run
#SBATCH --output=slurm_outputs/dsm_embedding_%j.log
#SBATCH --error=slurm_outputs/dsm_embedding_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xu.a@wehi.edu.au

module load python

source .env/bin/activate
python dsm_embedding_script.py
deactivate
