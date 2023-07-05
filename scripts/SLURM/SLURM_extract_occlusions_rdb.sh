#!/bin/bash
#
# This script is used to extract occlusions from the OpenDrive maps and to preprocess the data for the rdb scenarios

#SBATCH --job-name=preprocess_rdbs
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-24:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --array=0-267%20

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

rdb7_starts_at=242
rdb6_starts_at=197
rdb5_starts_at=183
rdb4_starts_at=160
rdb3_starts_at=132
rdb2_starts_at=102 # rdb1 has 102 episodes
rdb1_starts_at=0

if [ "$SLURM_ARRAY_TASK_ID" -ge $rdb7_starts_at ]; then
    scenario="rdb7"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb7_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb6_starts_at ]; then
    scenario="rdb6"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb6_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb5_starts_at ]; then
    scenario="rdb5"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb5_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb4_starts_at ]; then
    scenario="rdb4"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb4_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb3_starts_at ]; then
    scenario="rdb3"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb3_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb2_starts_at ]; then
    scenario="rdb2"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb2_starts_at))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb1_starts_at ]; then
    scenario="rdb1"
    episode_idx=$((SLURM_ARRAY_TASK_ID - rdb1_starts_at))
fi

python ~/OGRIT/scripts/extract_occlusions_one_episode.py --scenario $scenario --episode_idx $episode_idx
python ~/OGRIT/scripts/preprocess_one_episode.py --scenario $scenario --episode_idx $episode_idx