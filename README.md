# OGRIT

1) Install IGP2, as specified on https://github.com/uoe-agents/IGP2.

2) Install OGRIT with pip: 
    ```
    cd OGRIT
    pip install -e .
    ```

3) Copy the data from the [inD](https://www.ind-dataset.com/) dataset into `OGRIT/scenarios/data/ind`, and from the [rounD](https://www.round-dataset.com/) dataset into `OGRIT/scenarios/data/round`.


Note: Run all the scripts below from the directory `OGRIT/`.

4) Extract the occlusions, Preprocess the data and Extract the base and indicator features:
Extracting the indicator features for all the samples may take hours to complete.
    ```
    python scripts/extract_occlusions.py
    python scripts/preprocess_data.py --extract_indicator_features
    ```

5) Train OGRIT and the baseline (G-GRIT). Then calculate the evaluation metrics on the test set:

    ```
    python scripts/train_occlusion_grit.py
    python scripts/train_generalised_decision_trees.py
    python scripts/evaluate_models_from_features.py --models occlusion_grit,generalised_grit,occlusion_baseline
    python scripts/plot_results.py
    ```