#!/bin/bash
#
#SBATCH --job-name=OGRIT_EXTRACT_FRAMES_ROUND
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2G
#
#
### You need to change the array indices depending on the scenario.
### For bendplatz and frankenburg use: --array=0-10
### For heckstrasse use: --array=0-2
### For round leave as it is.
#
#SBATCH --array=0-21
#
#SBATCH --nice

### Adapt the following lines to reflect the paths on your server.
### To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

### Change "frankenburg" to be the scenario you want the features for. Choose one of: bendplatz, frankenburg, heckstrasse, round
python ~/OGRIT/scripts/preprocess_one_episode.py --scenario "frankenburg" --episode_idx $SLURM_ARRAY_TASK_ID