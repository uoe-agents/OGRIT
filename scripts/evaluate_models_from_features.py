import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle

from ogrit.core.base import get_base_dir, get_all_scenarios, get_data_dir, set_working_dir
from ogrit.core.data_processing import get_dataset
from ogrit.decisiontree.dt_goal_recogniser import Grit, GeneralisedGrit, UniformPriorGrit, OcclusionGrit, \
    OcclusionBaseline, NoPossiblyMissingFeaturesGrit, SpecializedOgrit
from ogrit.evaluation.model_evaluation import evaluate_models

from ogrit.goalrecognition.goal_recognition import PriorBaseline, UniformPriorBaseline


def drop_low_sample_agents(dataset, min_samples=2):
    unique_agent_pairs = dataset[['episode', 'agent_id', 'ego_agent_id', 'true_goal']].drop_duplicates()
    unique_agent_pairs['frame_count'] = 0
    unique_samples = dataset[['episode', 'agent_id', 'ego_agent_id', 'true_goal', 'frame_id']].drop_duplicates()
    vc = unique_samples.value_counts(['episode', 'agent_id', 'ego_agent_id'])
    vc = vc.to_frame().reset_index().rename(columns={0: 'sample_count'})
    new_dataset = dataset.merge(vc, on=['episode', 'agent_id', 'ego_agent_id'])
    new_dataset = new_dataset.loc[new_dataset.sample_count >= min_samples]
    return new_dataset


def main():
    parser = argparse.ArgumentParser(description='Train decision trees for goal recognition')

    parser.add_argument('--scenarios', type=str, help='List of scenarios, comma separated', default=None)
    parser.add_argument('--dataset', type=str, help='Subset of data to evaluate on', default='test')
    parser.add_argument('--models', type=str, help='List of models, comma separated', default='occlusion_grit')

    args = parser.parse_args()

    if args.scenarios is None:
        scenario_names = get_all_scenarios()
    else:
        scenario_names = args.scenarios.split(',')

    model_names = args.models.split(',')

    evaluate_models(scenario_names, model_names, args.dataset)


if __name__ == '__main__':
    set_working_dir()
    main()
