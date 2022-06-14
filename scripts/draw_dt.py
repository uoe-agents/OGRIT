from grit.core.base import get_img_dir
from grit.decisiontree.dt_goal_recogniser import OcclusionGrit

model = OcclusionGrit.load('heckstrasse')
truncate = ['RT', 'RFTT', 'RFFF']
goal_type = 'exit-roundabout'

goal_tree = model.decision_trees[goal_type]
pydot_tree = goal_tree.pydot_tree(truncate_edges=truncate)
pydot_tree.write_pdf(get_img_dir() + f'{model.get_model_name()}_{goal_type}.pdf')
