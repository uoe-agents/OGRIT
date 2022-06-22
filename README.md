# GRIT

Install OGRIT with pip: 
```
cd OGRIT
pip install -e .
```
Install IGP2, as specified on https://github.com/uoe-agents/IGP2.

Copy the data from the [inD](https://www.ind-dataset.com/) dataset into `OGRIT/scenarios/data/ind`, and from the [rounD](https://www.round-dataset.com/) dataset into `OGRIT/scenarios/data/round`.


Run all the scripts from the directory `OGRIT/`.

Extract the occlusions, Preprocess the data and Extract the base and indicator features:

```
python script/extract_occlusions.py
python scripts/preprocess_data.py --extract_indicator_features
```

Train OGRIT and the baseline. Then calculate the evaluation metrics on the test set:

```
python script/train_occlusion_grit.py
python script/train_generalised_grit.py
python script/evaluate_models_from_features.py --models occlusion_grit,generalised_grit,occlusion_baseline
python script/plot_results.py
```
