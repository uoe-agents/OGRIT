#!/bin/bash
#
# This script is used to preprocess the data for the RDB1 scenario.

#SBATCH --job-name=preprocess_rdbs
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-102%20

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

python ~/OGRIT/scripts/preprocess_one_episode.py --scenario "rdb1" --episode_idx $SLURM_ARRAY_TASK_ID

