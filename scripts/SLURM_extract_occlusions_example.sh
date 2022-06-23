#!/bin/bash
#
#SBATCH --job-name=OGRIT_EXTRACT_FRAMES_ROUND
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#
#SBATCH --array=0-21
#SBATCH --nice

source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/
python ~/OGRIT/scripts/preprocess_one_episode.py --scenario "round" --episode_idx $SLURM_ARRAY_TASK_ID