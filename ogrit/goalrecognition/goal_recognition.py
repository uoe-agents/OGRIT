import numpy as np
import pandas as pd
from scipy.special import xlogy

from igp2.data.scenario import ScenarioConfig
from igp2.opendrive.map import Map
from igp2.trajectory import VelocityTrajectory

from ogrit.core.base import get_data_dir, get_base_dir
from ogrit.core.feature_extraction import FeatureExtractor
from ogrit.goalrecognition.metrics import entropy


class GoalRecogniser:

    def goal_likelihood_from_features(self, features, goal_type, goal):
        raise NotImplementedError

    def batch_goal_probabilities(self, dataset):
        """

        Args:
            dataset: DataFrame with columns:
                path_to_goal_length,in_correct_lane,speed,acceleration,angle_in_lane,vehicle_in_front_dist,
                vehicle_in_front_speed,oncoming_vehicle_dist,goal_type,agent_id,possible_goal,true_goal,true_goal_type,
                frame_id,initial_frame_id,fraction_observed

        Returns:

        """
        dataset = dataset.copy()
        if 'ego_agent_id' not in dataset.columns:
            dataset['ego_agent_id'] = 0

        model_likelihoods = []

        for index, row in dataset.iterrows():
            features = row[list(FeatureExtractor.feature_names) + FeatureExtractor.indicator_features]
            goal_type = row['goal_type']
            goal = row['possible_goal']
            model_likelihood = self.goal_likelihood_from_features(features, goal_type, goal)

            model_likelihoods.append(model_likelihood)
        dataset['model_likelihood'] = model_likelihoods
        unique_samples = dataset[['episode', 'agent_id', 'ego_agent_id', 'frame_id', 'true_goal',
                                  'true_goal_type', 'fraction_observed']].drop_duplicates()

        model_predictions = []
        predicted_goal_types = []
        model_probs = []
        min_probs = []
        max_probs = []
        model_entropys = []
        model_norm_entropys = []
        true_goal_probs = []
        cross_entropies = []

        for index, row in unique_samples.iterrows():
            indices = ((dataset.episode == row.episode)
                       & (dataset.agent_id == row.agent_id)
                       & (dataset.ego_agent_id == row.ego_agent_id)
                       & (dataset.frame_id == row.frame_id))
            goals = dataset.loc[indices][['possible_goal', 'goal_type', 'model_likelihood']]

            if isinstance(self.goal_priors, pd.DataFrame):
                goals = goals.merge(self.goal_priors, 'left', left_on=['possible_goal', 'goal_type'],
                                    right_on=['true_goal', 'true_goal_type'])
            else:
                # use uniform prior for now
                num_goal_types = goals.possible_goal.unique().shape[0]
                goals['prior'] = 1.0 / num_goal_types

            goals['model_prob'] = goals.model_likelihood * goals.prior
            goals['model_prob'] = goals.model_prob / goals.model_prob.sum()
            idx = goals['model_prob'].idxmax()

            goal_prob_entropy = entropy(goals.model_prob)
            uniform_entropy = entropy(np.ones(goals.model_prob.shape[0])
                                      / goals.model_prob.shape[0])
            if uniform_entropy == 0:
                norm_entropy = 0
            else:
                norm_entropy = goal_prob_entropy / uniform_entropy
            model_prediction = goals['possible_goal'].loc[idx]
            predicted_goal_type = goals['goal_type'].loc[idx]
            predicted_goal_types.append(predicted_goal_type)
            model_predictions.append(model_prediction)
            model_prob = goals['model_prob'].loc[idx]
            true_goal_prob_df = goals['model_prob'].loc[goals.possible_goal == row.true_goal]
            if true_goal_prob_df.shape[0] == 0:
                true_goal_prob = 0
            else:
                true_goal_prob = float(goals['model_prob'].loc[goals.possible_goal == row.true_goal])
            cross_entropy = -(xlogy(goals.possible_goal == row.true_goal, goals.model_prob)).mean()
            max_prob = goals.model_prob.max()
            min_prob = goals.model_prob.min()
            max_probs.append(max_prob)
            min_probs.append(min_prob)
            model_probs.append(model_prob)
            model_entropys.append(goal_prob_entropy)
            model_norm_entropys.append(norm_entropy)
            true_goal_probs.append(true_goal_prob)
            cross_entropies.append(cross_entropy)

        unique_samples['model_prediction'] = model_predictions
        unique_samples['predicted_goal_type'] = predicted_goal_types
        unique_samples['model_probs'] = model_probs
        unique_samples['max_probs'] = max_probs
        unique_samples['min_probs'] = min_probs
        unique_samples['model_entropy'] = model_entropys
        unique_samples['model_entropy_norm'] = model_norm_entropys
        unique_samples['true_goal_prob'] = true_goal_probs
        unique_samples['cross_entropy'] = cross_entropies
        return unique_samples

    @classmethod
    def load(cls, scenario_name, data_dir=None):
        raise NotImplementedError


class FixedGoalRecogniser(GoalRecogniser):

    def __init__(self, goal_priors, scenario_map, goal_locs):
        self.goal_priors = goal_priors
        self.feature_extractor = FeatureExtractor(scenario_map)
        self.scenario_map = scenario_map
        self.goal_locs = goal_locs

    def goal_likelihood(self, goal_idx, frames, goal, agent_id):
        raise NotImplementedError

    def goal_likelihood_from_features(self, features, goal_type, goal):
        raise NotImplementedError

    def goal_probabilities(self, frames, agent_id):
        state_history = [f[agent_id] for f in frames]
        trajectory = VelocityTrajectory.from_agent_states(state_history)
        typed_goals = self.feature_extractor.get_typed_goals(trajectory, self.goal_locs)
        goal_probs = []
        for goal_idx, typed_goal in enumerate(typed_goals):
            if typed_goal is None:
                goal_prob = 0
            else:
                route = typed_goal.lane_path
                # get un-normalised "probability"
                prior = self.get_goal_prior(goal_idx, route)
                if prior == 0:
                    goal_prob = 0
                else:
                    likelihood = self.goal_likelihood(goal_idx, frames, typed_goal, agent_id)
                    goal_prob = likelihood * prior
            goal_probs.append(goal_prob)
        goal_probs = np.array(goal_probs)
        goal_probs = goal_probs / goal_probs.sum()
        return goal_probs

    @classmethod
    def load(cls, scenario_name, data_dir=None):
        priors = cls.load_priors(scenario_name, data_dir)
        scenario_map = Map.parse_from_opendrive(get_base_dir() + f"/scenarios/maps/{scenario_name}.xodr")
        scenario_config = ScenarioConfig.load(get_base_dir() + f"/scenarios/configs/{scenario_name}.json")
        return cls(priors, scenario_map, scenario_config.goals)

    @staticmethod
    def load_priors(scenario_name, data_dir=None):
        if data_dir is None:
            data_dir = get_data_dir()
        return pd.read_csv(data_dir + scenario_name + '_priors.csv')

    def get_goal_prior(self, goal_idx, route):
        goal_type = self.feature_extractor.goal_type(route)
        prior_series = self.goal_priors.loc[(self.goal_priors.true_goal == goal_idx) & (self.goal_priors.true_goal_type == goal_type)].prior
        if prior_series.shape[0] == 0:
            return 0
        else:
            return float(prior_series)


class PriorBaseline(FixedGoalRecogniser):

    def goal_likelihood(self, goal_idx, frames, route, agent_id):
        return 0.5

    def goal_likelihood_from_features(self, features, goal_type, goal):
        return 0.5


class UniformPriorBaseline(PriorBaseline):

    def __init__(self, goal_priors, scenario_map, goal_locs):
        super().__init__(goal_priors, scenario_map, goal_locs)
        self.goal_priors['prior'] = 1.0 / self.goal_priors.shape[0]
