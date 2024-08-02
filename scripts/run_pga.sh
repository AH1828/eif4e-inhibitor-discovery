#!/bin/bash
#SBATCH --job-name=RUN_PGA_100K
#SBATCH --ntasks-per-node=96
#SBATCH --nodes=1
#SBATCH -p extended-96core
#SBATCH --time=7-00:00:00
#SBATCH --out=test-out.%j
#SBATCH --error=test-err.%j
cd $HOME
source eif4e-inhibitor-discovery/.venv/bin/activate
python eif4e-inhibitor-discovery/src/run_PGA.py
