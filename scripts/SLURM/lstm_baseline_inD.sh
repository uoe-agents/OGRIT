#!/bin/bash

# This file is a template for the LSTM baseline experiments. It is meant to be used with the SLURM scheduler.
# It is used to compute the LSTM baseline for all scenarios and all input types.

#SBATCH --job-name=OGRIT_LSTM_BASELINE
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=10G
#
#SBATCH --array=0-12%6
#SBATCH --nice

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

neuweiler_starts_at=9
bendplatz_starts_at=6
frankenburg_starts_at=3
heckstrasse_starts_at=0


if [ "$SLURM_ARRAY_TASK_ID" -ge $neuweiler_starts_at ]; then
    train_scenarios="neuweiler"
    test_scenarios="neuweiler"

    ((start_3 = "neuweiler_starts_at+2"))
    ((start_2 = "neuweiler_starts_at+1"))

    if [ $SLURM_ARRAY_TASK_ID -ge "$start_3" ]; then
      input_type="absolute_position"
    elif [ "$SLURM_ARRAY_TASK_ID" -ge "$start_2" ]; then
      input_type="ogrit_features"
    else
      input_type="relative_position"
    fi

elif [ "$SLURM_ARRAY_TASK_ID" -ge $bendplatz_starts_at ]; then
    train_scenarios="bendplatz"
    test_scenarios="bendplatz"

    ((start_3 = "bendplatz_starts_at+2"))
    ((start_2 = "bendplatz_starts_at+1"))

    if [ "$SLURM_ARRAY_TASK_ID" -ge "$start_3" ]; then
      input_type="absolute_position"
    elif [ "$SLURM_ARRAY_TASK_ID" -ge "$start_2" ]; then
      input_type="ogrit_features"
    else
      input_type="relative_position"
    fi

elif [ $SLURM_ARRAY_TASK_ID -ge $frankenburg_starts_at ]; then
    train_scenarios="frankenburg"
    test_scenarios="frankenburg"

    ((start_3 = "frankenburg_starts_at+2"))
    ((start_2 = "frankenburg_starts_at+1"))

    if [ "$SLURM_ARRAY_TASK_ID" -ge "$start_3" ]; then
      input_type="absolute_position"
    elif [ "$SLURM_ARRAY_TASK_ID" -ge "$start_2" ]; then
      input_type="ogrit_features"
    else
      input_type="relative_position"
    fi

else
    train_scenarios="heckstrasse"
    test_scenarios="heckstrasse"

    ((start_3 = "heckstrasse_starts_at+2"))
    ((start_2 = "heckstrasse_starts_at+1"))

    if [ "$SLURM_ARRAY_TASK_ID" -ge "$start_3" ]; then
      input_type="absolute_position"
    elif [ "$SLURM_ARRAY_TASK_ID" -ge "$start_2" ]; then
      input_type="ogrit_features"
    else
      input_type="relative_position"
    fi
fi


python ~/OGRIT/baselines/lstm/get_results.py --train_scenarios $train_scenarios --test_scenarios $test_scenarios --input_type $input_type

