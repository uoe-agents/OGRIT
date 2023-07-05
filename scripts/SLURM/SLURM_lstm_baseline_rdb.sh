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
#SBATCH --array=0-21%6
#SBATCH --nice

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

rdb7_starts_at=18
rdb6_starts_at=15
rdb5_starts_at=12
rdb4_starts_at=9
rdb3_starts_at=6
rdb2_starts_at=3 # There are 3 lstms for rdb1
rdb1_starts_at=0


if [ "$SLURM_ARRAY_TASK_ID" -ge $rdb7_starts_at ]; then
    train_scenarios="rdb7"
    test_scenarios="rdb7"

    ((start_3 = "rdb7_starts_at+2"))
    ((start_2 = "rdb7_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb6_starts_at ]; then
    train_scenarios="rdb6"
    test_scenarios="rdb6"

    ((start_3 = "rdb6_starts_at+2"))
    ((start_2 = "rdb6_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb5_starts_at ]; then
    train_scenarios="rdb5"
    test_scenarios="rdb5"

    ((start_3 = "rdb5_starts_at+2"))
    ((start_2 = "rdb5_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb4_starts_at ]; then
    train_scenarios="rdb4"
    test_scenarios="rdb4"

    ((start_3 = "rdb4_starts_at+2"))
    ((start_2 = "rdb4_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb3_starts_at ]; then
    train_scenarios="rdb3"
    test_scenarios="rdb3"

    ((start_3 = "rdb3_starts_at+2"))
    ((start_2 = "rdb3_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb2_starts_at ]; then
    train_scenarios="rdb2"
    test_scenarios="rdb2"

    ((start_3 = "rdb2_starts_at+2"))
    ((start_2 = "rdb2_starts_at+1"))

elif [ "$SLURM_ARRAY_TASK_ID" -ge $rdb1_starts_at ]; then
    train_scenarios="rdb1"
    test_scenarios="rdb1"

    ((start_3 = "rdb1_starts_at+2"))
    ((start_2 = "rdb1_starts_at+1"))
fi

if [ $SLURM_ARRAY_TASK_ID -ge "$start_3" ]; then
      input_type="absolute_position"
    elif [ "$SLURM_ARRAY_TASK_ID" -ge "$start_2" ]; then
      input_type="ogrit_features"
    else
      input_type="relative_position"
    fi


python ~/OGRIT/baselines/lstm/get_results.py --train_scenarios $train_scenarios --test_scenarios $test_scenarios --input_type $input_type

