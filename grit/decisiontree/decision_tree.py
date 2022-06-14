import pickle
from typing import Union

import numpy as np
import pandas as pd
from scipy.special import xlogy

pd.options.mode.chained_assignment = None

import pydot
from sklearn.tree import _tree

from grit.core.feature_extraction import FeatureExtractor


class Decision:

    def __init__(self, feature_name, true_child, false_child):
        self.feature_name = feature_name
        self.true_child = true_child
        self.false_child = false_child

    def rule(self, features):
        raise NotImplementedError

    def select_child(self, features):
        if self.rule(features):
            return self.true_child
        else:
            return self.false_child


class BinaryDecision(Decision):

    def rule(self, features):
        return features[self.feature_name]

    def __str__(self):
        return self.feature_name + '\n'


class ThresholdDecision(Decision):

    def __init__(self, threshold, *args):
        super().__init__(*args)
        self.threshold = threshold

    def rule(self, features):
        return features[self.feature_name] > self.threshold

    def __str__(self):

        if (self.feature_name in FeatureExtractor.indicator_features
                or FeatureExtractor.feature_names[self.feature_name] == 'binary'):
            return self.feature_name + '\n'
        elif FeatureExtractor.feature_names[self.feature_name] == 'integer':
            return '{} > {}\n'.format(self.feature_name, int(self.threshold))
        else:
            return '{} > {:.2f}\n'.format(self.feature_name, self.threshold)


class Node:
    def __init__(self, value, decision=None, level=0):
        self.value = value
        self.decision = decision
        self.counts = [None, None]
        self.reached = False
        self.level = level

    def traverse(self, features, terminate_on_missing=False):
        self.reached = True
        current_node = self
        while current_node.decision is not None:
            if (terminate_on_missing
                    and current_node.decision.feature_name in FeatureExtractor.possibly_missing_features
                    and features[FeatureExtractor.possibly_missing_features[current_node.decision.feature_name]]):
                # feature is missing
                return current_node.value
            current_node = current_node.decision.select_child(features)
            current_node.reached = True
        return current_node.value

    def reset_reached(self):
        self.reached = False
        if self.decision is not None:
            self.decision.true_child.reset_reached()
            self.decision.false_child.reset_reached()

    def __str__(self):
        text = ''
        text += '{0:.3f} {1}\n'.format(self.value, self.counts)
        # text += '{0:.3f}\n'.format(self.value)
        if self.decision is not None:
            text += str(self.decision)
        return text

    def get_text(self, show_counts=False):
        text = ''

        if self.decision is not None:
            text += str(self.decision)
        else:
            if show_counts:
                text += '{0:.3f} {1}'.format(self.value, self.counts)
            else:
                text += '{0:.3f}'.format(self.value)
        return text

    @classmethod
    def from_sklearn(cls, input_tree, feature_types):
        # based on:
        # https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

        tree_ = input_tree.tree_
        feature_names = [*feature_types]
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node):
            value = tree_.value[node][0][1] / tree_.value[node].sum()
            out_node = Node(value)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                true_child = recurse(tree_.children_right[node])
                false_child = recurse(tree_.children_left[node])
                if feature_types[name] in ['scalar', 'integer']:
                    out_node.decision = ThresholdDecision(threshold, name, true_child, false_child)
                elif feature_types[name] == 'binary':
                    out_node.decision = BinaryDecision(name, true_child, false_child)
                else:
                    raise ValueError('invalid feature type')
            return out_node

        return recurse(0)

    def set_values(self, samples: pd.DataFrame, goal: Union[int, str], alpha=0):
        # check if we are using generalised goal trees (one tree per goal type)
        possible_goal = samples.goal_type if isinstance(goal, str) else samples.possible_goal
        samples['has_goal'] = samples.possible_goal == samples.true_goal
        goal_training_samples = samples.loc[possible_goal == goal]

        N = goal_training_samples.shape[0]
        Ng = goal_training_samples.has_goal.sum()
        goal_normaliser = (N + 2 * alpha) / 2 / (Ng + alpha)
        non_goal_normaliser = (N + 2 * alpha) / 2 / (N - Ng + alpha)
        feature_names = [*FeatureExtractor.feature_names]

        def recurse(node, node_samples):
            Nng = node_samples.loc[node_samples.has_goal].shape[0]
            Nn = node_samples.shape[0]
            Nng_norm = (Nng + alpha) * goal_normaliser
            Nn_norm = Nng_norm + (Nn - Nng + alpha) * non_goal_normaliser
            value = Nng_norm / Nn_norm
            node.value = value
            node.counts = [Nng, Nn - Nng]
            features = node_samples.loc[:, feature_names]
            if node.decision is not None:
                rule_true = node.decision.rule(features)
                true_child_samples = node_samples.loc[rule_true]
                false_child_samples = node_samples.loc[~rule_true]
                recurse(node.decision.true_child, true_child_samples)
                recurse(node.decision.false_child, false_child_samples)

        recurse(self, goal_training_samples)

    @classmethod
    def fit(cls, samples: pd.DataFrame, goal: Union[int, str], alpha=0, min_samples_leaf=1, max_depth=None,
            ccp_alpha=0.):
        possible_goal = samples.goal_type if isinstance(goal, str) else samples.possible_goal
        samples['has_goal'] = samples.possible_goal == samples.true_goal
        goal_training_samples = samples.loc[possible_goal == goal]

        N = goal_training_samples.shape[0]
        Ng = goal_training_samples.has_goal.sum()
        goal_normaliser = (N + 2 * alpha) / 2 / (Ng + alpha)
        non_goal_normaliser = (N + 2 * alpha) / 2 / (N - Ng + alpha)

        base_features = FeatureExtractor.feature_names.keys()
        possibly_missing_features = FeatureExtractor.possibly_missing_features
        indicator_features = FeatureExtractor.indicator_features

        def _recursive_split(node: Node, node_samples: pd.DataFrame, true_indicators, false_indicators):

            if (node_samples.has_goal.nunique() != 1
                    and node_samples.shape[0] > min_samples_leaf
                    and (max_depth is None or node.level < max_depth)):

                # find best decision
                best_impurity_decrease = 0
                best_feature = None
                best_threshold = None
                impurity = cls.cross_entropy(node_samples, goal_normaliser, non_goal_normaliser, alpha)

                Nn = node_samples.shape[0]
                Nng = node_samples.loc[node_samples.has_goal].shape[0]

                allowed_features = [f for f in (list(base_features) + indicator_features) if
                                    f not in possibly_missing_features
                                    or possibly_missing_features[f] in false_indicators]

                for feature in allowed_features:
                    # find best threshold
                    impurity_decrease, threshold = cls.get_best_threshold(
                        node_samples, feature, N, Nn, Nng, alpha, goal_normaliser, non_goal_normaliser, impurity)

                    if impurity_decrease > best_impurity_decrease:
                        best_impurity_decrease = impurity_decrease
                        best_feature = feature
                        best_threshold = threshold

                # look ahead by one node when considering indicator + possibly missing features
                unknown_missing_features = [f for f in possibly_missing_features if
                                            possibly_missing_features[f] not in (true_indicators + false_indicators)]
                for feature in unknown_missing_features:
                    indicator = possibly_missing_features[feature]
                    indicator_impurity_decrease, indicator_threshold = cls.get_best_threshold(
                        node_samples, indicator, N, Nn, Nng, alpha, goal_normaliser, non_goal_normaliser, impurity)
                    child_samples = node_samples.loc[~node_samples[indicator]]
                    child_impurity = cls.cross_entropy(child_samples, goal_normaliser, non_goal_normaliser, alpha)
                    child_impurity_decrease, _ = cls.get_best_threshold(
                        child_samples, feature, N, Nn, Nng, alpha, goal_normaliser, non_goal_normaliser,
                        child_impurity)

                    impurity_decrease = indicator_impurity_decrease + child_impurity_decrease - ccp_alpha
                    if impurity_decrease > best_impurity_decrease:
                        best_impurity_decrease = impurity_decrease
                        best_feature = indicator
                        best_threshold = 0.5

                if best_impurity_decrease > 0:
                    true_idx = node_samples[best_feature] > best_threshold
                    true_samples = node_samples.loc[true_idx]
                    false_samples = node_samples.loc[~true_idx]
                    true_child = cls.get_node(true_samples, node.level + 1, goal_normaliser,
                                              non_goal_normaliser, alpha)
                    false_child = cls.get_node(false_samples, node.level + 1, goal_normaliser,
                                               non_goal_normaliser, alpha)
                    node.decision = ThresholdDecision(best_threshold, best_feature, true_child, false_child)
                    true_idx = node.decision.rule(node_samples)
                    if best_feature in indicator_features:
                        true_child_true_indicators = true_indicators + [best_feature]
                        false_child_false_indicators = false_indicators + [best_feature]
                    else:
                        true_child_true_indicators = true_indicators
                        false_child_false_indicators = false_indicators
                    _recursive_split(node.decision.true_child, node_samples.loc[true_idx],
                                     true_child_true_indicators, false_indicators)
                    _recursive_split(node.decision.false_child, node_samples.loc[~true_idx],
                                     true_indicators, false_child_false_indicators)
            return node

        root = cls.get_node(goal_training_samples, 0, goal_normaliser, non_goal_normaliser, alpha)
        _recursive_split(root, goal_training_samples, [], [])

        if ccp_alpha > 0:
            root.prune(goal_training_samples, N, ccp_alpha, goal_normaliser, non_goal_normaliser, alpha)

        return root

    @staticmethod
    def get_best_threshold(node_samples, feature, N, Nn, Nng, alpha, goal_normaliser, non_goal_normaliser,
                           impurity):

        df = node_samples[[feature, 'has_goal']].sort_values(feature)
        df['Nnt'] = np.arange(1, df.shape[0] + 1)
        df['Nng_true'] = df.has_goal.cumsum()
        df.drop_duplicates(feature, inplace=True, keep='last')
        if df.shape[0] < 2:
            return 0, None
        df['Nnf'] = Nn - df.Nnt
        df['Nng_false'] = Nng - df.Nng_true
        df['threshold'] = df[feature].rolling(2).mean().shift(-1)
        df = df[:-1]

        df['pg_true'] = (df.Nng_true + alpha) / (df.Nnt + 2 * alpha)
        df['png_true'] = 1 - df.pg_true

        df['pg_false'] = (df.Nng_false + alpha) / (df.Nnf + 2 * alpha)
        df['png_false'] = 1 - df.pg_false

        df['impurity_true'] = (
                - goal_normaliser * xlogy(df.pg_true, df.pg_true)
                - non_goal_normaliser * xlogy(df.png_true, df.png_true))
        df['impurity_false'] = (
                - goal_normaliser * xlogy(df.pg_false, df.pg_false)
                - non_goal_normaliser * xlogy(df.png_false, df.png_false))
        df['impurity_decrease'] = Nn / N * (
                impurity - df.Nnt / Nn * df.impurity_true
                - df.Nnf / Nn * df.impurity_false)
        df.reset_index(inplace=True)
        best = df.loc[df.impurity_decrease.idxmax(), :]
        impurity_decrease = float(best.impurity_decrease)
        threshold = float(best.threshold)
        return impurity_decrease, threshold

    def post_prune(self, samples: pd.DataFrame, goal: Union[int, str], alpha=0, min_samples_leaf=1, max_depth=None,
            ccp_alpha=0.):
        possible_goal = samples.goal_type if isinstance(goal, str) else samples.possible_goal
        samples['has_goal'] = samples.possible_goal == samples.true_goal
        goal_training_samples = samples.loc[possible_goal == goal]
        N = goal_training_samples.shape[0]
        Ng = goal_training_samples.has_goal.sum()
        goal_normaliser = (N + 2 * alpha) / 2 / (Ng + alpha)
        non_goal_normaliser = (N + 2 * alpha) / 2 / (N - Ng + alpha)
        self.prune(goal_training_samples, N, ccp_alpha, goal_normaliser, non_goal_normaliser, ccp_alpha, alpha)

    def prune(self, node_samples: pd.DataFrame, total_samples: int, ccp_alpha=0., goal_normaliser=1.,
              non_goal_normaliser=1., alpha=1.):

        if self.decision is not None:
            impurity = self.cross_entropy(node_samples, goal_normaliser, non_goal_normaliser)
            true_idx = self.decision.rule(node_samples)
            true_samples = node_samples.loc[true_idx]
            false_samples = node_samples.loc[~true_idx]

            self.decision.true_child.prune(true_samples, total_samples, ccp_alpha, goal_normaliser,
                                           non_goal_normaliser, alpha)
            self.decision.false_child.prune(false_samples, total_samples, ccp_alpha, goal_normaliser,
                                            non_goal_normaliser, alpha)

            if self.decision.true_child.decision is None and self.decision.true_child.decision is None:
                true_impurity = self.cross_entropy(true_samples, goal_normaliser, non_goal_normaliser, alpha)
                false_impurity = self.cross_entropy(false_samples, goal_normaliser, non_goal_normaliser, alpha)

                Nn = node_samples.shape[0]
                Nnt = true_samples.shape[0]
                Nnf = false_samples.shape[0]
                impurity_decrease = Nn / total_samples * (impurity - Nnt / Nn * true_impurity
                                                                   - Nnf / Nn * false_impurity)
                if impurity_decrease <= ccp_alpha:
                    self.decision = None

    @staticmethod
    def cross_entropy(samples: pd.DataFrame, goal_normaliser=1., non_goal_normaliser=1., alpha=0.) -> float:
        Nng = samples.loc[samples.has_goal].shape[0]
        Nn = samples.shape[0]
        pg = (Nng + alpha) / (Nn + 2 * alpha)
        png = 1 - pg
        return - goal_normaliser * xlogy(pg, pg) - non_goal_normaliser * xlogy(png, png)

    @classmethod
    def get_node(cls, node_samples: pd.DataFrame, level, goal_normaliser: float, non_goal_normaliser: float, alpha=0.):
        Nng = node_samples.loc[node_samples.has_goal].shape[0]
        Nn = node_samples.shape[0]
        Nng_norm = (Nng + alpha) * goal_normaliser
        Nn_norm = Nng_norm + (Nn - Nng + alpha) * non_goal_normaliser
        value = Nng_norm / Nn_norm
        #value = Nng / Nn
        node = cls(value, level=level)
        node.counts = [Nng, Nn - Nng]
        return node

    def pydot_tree(self, truncate_edges=None):
        if truncate_edges is None:
            truncate_edges = []

        graph = pydot.Dot(graph_type='digraph')

        def recurse(graph, root, idx='R'):

            if root.decision is None:
                shape = 'oval'
                style = 'filled'
                color = '#7cb571'
                fillcolor = '#d2e8d5'
            elif root.decision.feature_name in FeatureExtractor.indicator_features:
                shape = 'oval'
                style = 'filled'
                color = '#d06769'
                fillcolor = '#ffcccd'
            elif root.decision.feature_name in FeatureExtractor.possibly_missing_features:
                shape = 'octagon'
                color = '#82aacc'
                style = 'filled'
                fillcolor = '#d5e9fb'
            else:
                shape = 'box'
                color = 'black'
                style = 'solid'
                fillcolor = 'white'

            if root.reached:
                style = 'filled'
                color = '#5ebfad'

            node = pydot.Node(idx, label=root.get_text(), shape=shape, style=style, color=color, fillcolor=fillcolor)
            graph.add_node(node)
            if root.decision is not None:
                true_idx = idx + 'T'
                true_weight = root.decision.true_child.value / root.value
                if true_idx in truncate_edges:
                    dummy_child = pydot.Node(true_idx, label=' ', color='white')
                    graph.add_node(dummy_child)
                    graph.add_edge(pydot.Edge(node, dummy_child, style='dashed',
                                              label='T: {:.2f}'.format(true_weight)))
                else:
                    true_child = recurse(graph, root.decision.true_child, true_idx)
                    graph.add_edge(pydot.Edge(node, true_child, label='T: {:.2f}'.format(true_weight)))

                false_idx = idx + 'F'
                false_weight = root.decision.false_child.value / root.value
                if false_idx in truncate_edges:
                    dummy_child = pydot.Node(false_idx, label=' ', color='white')
                    graph.add_node(dummy_child)
                    graph.add_edge(pydot.Edge(node, dummy_child, style='dashed',
                                              label='F: {:.2f}'.format(false_weight)))
                else:
                    false_child = recurse(graph, root.decision.false_child, false_idx)
                    graph.add_edge(pydot.Edge(node, false_child, label='F: {:.2f}'.format(false_weight)))
            return node

        recurse(graph, self)
        return graph

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
