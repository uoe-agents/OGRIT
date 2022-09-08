import pandas as pd

from ogrit.core.data_processing import get_multi_scenario_dataset
from ogrit.core.feature_extraction import FeatureExtractor

scenario_names = ['neuweiler']# ['heckstrasse', 'bendplatz', 'frankenburg']
dataset = get_multi_scenario_dataset(scenario_names, subset='all')

pd.set_option('display.max_columns', None)
#print(dataset.columns)
print(dataset[['exit_number_missing', 'scenario']
      ].groupby('scenario').mean())
print(dataset[FeatureExtractor.indicator_features + ['scenario']
      ].groupby('scenario').mean())
