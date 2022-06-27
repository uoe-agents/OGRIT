from ogrit.core.base import get_img_dir
from ogrit.decisiontree.dt_goal_recogniser import OcclusionGrit, GeneralisedGrit

model = OcclusionGrit.load('heckstrasse')
model.save_images()
# truncate = ['RT', 'RFTT', 'RFFF']
truncate = []
goal_type = 'turn-right'

goal_tree = model.decision_trees[goal_type]

pydot_tree = goal_tree.pydot_tree(truncate_edges=truncate)
pydot_tree.write_pdf(get_img_dir() + f'{model.get_model_name()}_{goal_type}.pdf')
