#!/bin/bash
#
# This script is used to get the results for the lstm generalization experiments.

#SBATCH --job-name=generalization_experiment
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-24:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --gres=gpu:1
#SBATCH --nice
#SBATCH --array=0-3%2

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

fill_occluded_frames_mode="remove"
update_hz=25

input_type="relative_position"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    variant_i_train_scenarios="rdb2,rdb3,rdb6,rdb7"
    variant_i_test_scenarios="neuweiler,rdb4,rdb5"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    variant_i_train_scenarios="rdb2,rdb3,rdb4,rdb6,rdb7"
    variant_i_test_scenarios="neuweiler,rdb5"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    variant_i_train_scenarios="rdb2,rdb3,rdb4,rdb5,rdb6,rdb7"
    variant_i_test_scenarios="neuweiler"
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    variant_i_train_scenarios="rdb1,rdb2,rdb3,rdb4,rdb5,rdb6,rdb7"
    variant_i_test_scenarios="neuweiler"
fi

python ~/OGRIT/baselines/lstm/get_results.py --recompute_dataset --train_scenarios $variant_i_train_scenarios --test_scenarios $variant_i_test_scenarios  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz