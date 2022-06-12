# GRIT

Things to do before running:
- add folder "data" to store csv with sample data
- add folder "occlusions" to store occlusion data
- add the ind and round datasets into "scenarios/data/ind" and "scenarios/data/round", respectively
- to extract base features only use prepare_episode_dataset(("bendplatz", 0, False)) else, if you also want the indicator features use prepare_episode_dataset(("bendplatz", 0, True))
This repo contains work in progess building on the paper:

["GRIT: Fast, Interpretable, and Verifiable Goal Recognition with Learned Decision Trees for Autonomous Driving"](https://arxiv.org/abs/2103.06113)
by Brewitt, et al. [1] accepted at IROS 2021
