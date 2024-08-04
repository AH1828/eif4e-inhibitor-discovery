#!/bin/bash
#SBATCH --job-name=AE_HP_TUNING
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1
#SBATCH -p debug-40core
#SBATCH --time=1:00:00
#SBATCH --out=test-out.%j
#SBATCH --error=test-err.%j
cd $HOME
source eif4e-inhibitor-discovery/.venv/bin/activate
python eif4e-inhibitor-discovery/src/autoencoder_hp.py
