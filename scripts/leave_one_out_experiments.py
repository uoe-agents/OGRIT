from pathlib import Path

from ogrit.core.base import get_data_dir, get_base_dir, get_predictions_dir
from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, SpecializedOgrit
from ogrit.evaluation.model_evaluation import evaluate_models

scenarios = ['heckstrasse', 'bendplatz', 'frankenburg']

predictions_dir = get_predictions_dir() + f'/loocv/'
Path(predictions_dir).mkdir(parents=True, exist_ok=True)

for test_scenario in scenarios:
    train_scenarios = scenarios.copy()
    train_scenarios.remove(test_scenario)

    training_set = get_multi_scenario_dataset(train_scenarios, 'all')

    models_dir = get_data_dir() + f'/loocv/{test_scenario}/'
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    ogrit = OcclusionGrit.train(train_scenarios,
                                dataset=training_set,
                                criterion='entropy',
                                min_samples_leaf=10,
                                max_depth=7,
                                alpha=1, ccp_alpha=0.0001,
                                balance_scenarios=True)
    ogrit.save(data_dir=models_dir)

    # evaluate
    evaluate_models(model_names=['occlusion_grit'], dataset_name='test', predictions_dir=predictions_dir,
                    models_dir=models_dir, scenario_names=[test_scenario], suffix='_loocv')


