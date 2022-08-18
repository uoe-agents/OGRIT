import pandas as pd

from ogrit.core.base import get_data_dir
from ogrit.core.data_processing import get_dataset

datasets = ['train', 'valid', 'test']
scenarios = ['heckstrasse', 'bendplatz', 'frankenburg']


num_samples = pd.DataFrame(index=datasets, columns=scenarios)

data_dir = get_data_dir() + "/occlusion_subset/"
#data_dir = None
for dataset_name in datasets:
    for scenario in scenarios:
        dataset = get_dataset(scenario, dataset_name, data_dir=data_dir)

        num_samples.loc[dataset_name, scenario] = dataset.shape[0]
print(num_samples)
