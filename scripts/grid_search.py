import pandas as pd
import numpy as np

from ogrit.core.data_processing import get_dataset
from ogrit.decisiontree.dt_goal_recogniser import GeneralisedGrit, OcclusionGrit
from ogrit.core.base import get_all_scenarios

scenario_names = get_all_scenarios()

ccp_values = [0.001, 0.0003, 0.0001]
true_goal_prob = []

for idx, ccp_alpha in enumerate(ccp_values):
    print(f'{idx+1}/{len(ccp_values)}')

    model = OcclusionGrit.train(scenario_names,
                                 criterion='entropy',
                                 min_samples_leaf=10,
                                 max_depth=7,
                                 alpha=1, ccp_alpha=ccp_alpha)
    alpha_true_goal_prob = []
    for scenario_name in scenario_names:
        dataset = get_dataset(scenario_name, 'valid')
        unique_samples = model.batch_goal_probabilities(dataset)
        unique_samples['model_correct'] = (unique_samples['model_prediction']
                                           == unique_samples['true_goal'])
        alpha_true_goal_prob.append(unique_samples.true_goal_prob.mean())
    true_goal_prob.append(np.mean(alpha_true_goal_prob))

print(pd.DataFrame({'ccp_values': ccp_values, 'true goal prob': true_goal_prob}))


