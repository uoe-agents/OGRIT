#!/bin/bash
#SBATCH --job-name=GRIT
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=1G

source ~/miniconda3/etc/profile.d/conda.sh
conda activate igp2env
export PYTHONPATH=$HOME/igp2-dev
python ~/GRIT-OpenDrive/core/data_processing.py --workers $SBATCH_NUM_PROC --scenario round
