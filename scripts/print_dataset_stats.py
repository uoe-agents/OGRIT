import pandas as pd

from ogrit.core.data_processing import get_dataset

datasets = ['train', 'valid', 'test']
scenarios = ['neuweiler']


num_samples = pd.DataFrame(index=datasets, columns=scenarios)

for dataset_name in datasets:
    for scenario in scenarios:
        dataset = get_dataset(scenario, dataset_name)

        num_samples.loc[dataset_name, scenario] = dataset.shape[0]
print(num_samples)
