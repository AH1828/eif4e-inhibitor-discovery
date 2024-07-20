#!/bin/bash
#SBATCH --job-name=DATA_PREPROCESSING
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH -p long-96core
#SBATCH --time=48:00:00
#SBATCH --out=test-out.%j
#SBATCH --error=test-err.%j
cd $HOME
source eif4e-inhibitor-discovery/.venv/bin/activate
python eif4e-inhibitor-discovery/src/data_preprocessing.py
