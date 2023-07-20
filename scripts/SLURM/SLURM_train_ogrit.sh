#!/bin/bash
#SBATCH --job-name=train_ogrit
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

python ~/OGRIT/scripts/train_occlusion_grit.py --scenarios rdb1,rdb2,rdb3,rdb4,rdb5,rdb6,rdb7 --suffix all_rdbs_angle_to_goal

# evalaute on Neuweiler
python ~/OGRIT/scripts/evaluate_models_from_features.py --scenarios neuweiler --models occlusion_grit_all_rdbs_angle_to_goal