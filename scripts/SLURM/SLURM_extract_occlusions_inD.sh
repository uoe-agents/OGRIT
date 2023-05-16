#!/bin/bash
#
# This script is used to extract occlusions from the OpenDrive maps and to preprocess the data for the Heckstrasse,
# Frankenburg, Bendplatz and Neuweiler scenarios.

#SBATCH --job-name=preprocess_all_scenarios
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-24:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --array=0-46%6

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

neuweiler_starts_at=25 # Heckstrasse has 3 episodes, Frankenburg has 11 episodes, Bendplatz has 11 episodesì, thus
                         # Neuweiler starts at 3 + 11 + 11 = 25.
bendplatz_starts_at=14
frankenburg_starts_at=3
heckstrasse_starts_at=0

if [ "$SLURM_ARRAY_TASK_ID" -ge $neuweiler_starts_at ]; then
    scenario="neuweiler"
    episode_idx=$((SLURM_ARRAY_TASK_ID - neuweiler_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $bendplatz_starts_at ]; then
    scenario="bendplatz"
    episode_idx=$((SLURM_ARRAY_TASK_ID - bendplatz_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $frankenburg_starts_at ]; then
    scenario="frankenburg"
    episode_idx=$((SLURM_ARRAY_TASK_ID - frankenburg_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $heckstrasse_starts_at ]; then
    scenario="heckstrasse"
    episode_idx=$((SLURM_ARRAY_TASK_ID - heckstrasse_starts_at))
fi

python ~/OGRIT/scripts/extract_occlusions_one_episode.py --scenario $scenario --episode_idx $episode_idx
python ~/OGRIT/scripts/preprocess_one_episode.py --scenario $scenario --episode_idx $episode_idx