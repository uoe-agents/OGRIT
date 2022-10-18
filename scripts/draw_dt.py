from ogrit.core.base import get_img_dir, get_data_dir
from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit, OcclusionBaseline, OgritOracle, \
    NoPossiblyMissingFeaturesOGrit, NoPossiblyMissingFeaturesGrit

models = [OcclusionGrit]
for model_class in models:
    test_scenario = 'heckstrasse'
    models_dir = get_data_dir() #+ f'/loocv/{test_scenario}/'
    model = model_class.load(test_scenario, episode_idx=0)
    model.save_images()
    # truncate = ['RT', 'RFTT', 'RFFF']
    truncate = []
    goal_type = 'exit-roundabout'

    for goal_type in model.decision_trees:
        goal_tree = model.decision_trees[goal_type]
        model_name = model.get_model_name()

        pydot_tree = goal_tree.pydot_tree(truncate_edges=truncate)
        pydot_tree.write_pdf(get_img_dir() + f'{model.get_model_name()}_{goal_type}.pdf')
