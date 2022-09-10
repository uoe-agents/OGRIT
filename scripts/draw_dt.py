from ogrit.core.base import get_img_dir, get_data_dir
from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit, OcclusionBaseline, OgritOracle, \
    NoPossiblyMissingFeaturesOGrit

models = [OcclusionGrit, NoPossiblyMissingFeaturesOGrit, OgritOracle]
for model_class in models:
    test_scenario = 'neuweiler'
    models_dir = get_data_dir() #+ f'/loocv/{test_scenario}/'
    model = model_class.load(test_scenario)
    model.save_images()
    # truncate = ['RT', 'RFTT', 'RFFF']
    truncate = []
    goal_type = 'turn-left'

    # goal_tree = model.decision_trees[goal_type]
    # model_name = model.get_model_name()
    # #model_name = 'ogrit_loocv'
    #
    # pydot_tree = goal_tree.pydot_tree(truncate_edges=truncate)
    # pydot_tree.write_pdf(get_img_dir() + f'{model.get_model_name()}_{goal_type}.pdf')
