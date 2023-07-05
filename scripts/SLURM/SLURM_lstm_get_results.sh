#!/bin/bash
#
# This script is used to get the results for each scenario when the training and test scenarios are the same. E.g.,
# when the training and test scenarios are both bendplatz.

#SBATCH --job-name=test_lstm
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-24:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --nice

# Adapt the following lines to reflect the paths on your server.
# To create a new environment, first run: `conda create --name OGRIT python=3.8`
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OGRIT

export PYTHONPATH=$HOME/OGRIT/

fill_occluded_frames_mode="remove"
update_hz=25

input_type="ogrit_features"

# inD and rounD datasets
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "bendplatz" --test_scenarios "bendplatz"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "frankenburg" --test_scenarios "frankenburg"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "heckstrasse" --test_scenarios "heckstrasse"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "neuweiler" --test_scenarios "neuweiler"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz

# OpenDD dataset
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb1" --test_scenarios "rdb1"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb2" --test_scenarios "rdb2"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb3" --test_scenarios "rdb3"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb4" --test_scenarios "rdb4"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb5" --test_scenarios "rdb5"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb6" --test_scenarios "rdb6"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
python ~/OGRIT/baselines/lstm/get_results.py --evaluate_only --train_scenarios "rdb7" --test_scenarios "rdb7"  --input_type $input_type --fill_occluded_frames_mode $fill_occluded_frames_mode --update_hz $update_hz
