import argparse

from ogrit.core.base import get_all_scenarios, set_working_dir
from ogrit.evaluation.model_evaluation import evaluate_models


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
