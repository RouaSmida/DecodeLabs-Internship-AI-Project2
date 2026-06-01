# Project 2: Data Classification Using AI

Build a basic classification model on a small dataset and extend it with practical improvements. The project uses built-in scikit-learn datasets (Iris by default), so there are no external data files to download.

## Goals and Requirements
- Load and understand a dataset
- Split data into training and testing sets
- Apply a simple classification algorithm
- Extend with stronger evaluation and model tuning

## Project Structure
```
Project2/
  README.md
  requirements.txt
  src/
    train.py
  outputs/
    (generated on run)
```

## Setup
1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Run
### Default training (KNN with tuned K)
```
python src/train.py
```

### Pick a dataset
```
python src/train.py --dataset wine
python src/train.py --dataset breast_cancer
```

### Tune by a different metric
```
python src/train.py --tune-metric f1_macro
```

### Save extra comparison models
```
python src/train.py --compare-models
```

### Change test size or random seed
```
python src/train.py --test-size 0.2 --random-state 42
```

### Predict from new data
Provide a .csv or .json with columns matching the dataset feature names. A sample file is included as `sample_inputs.csv`.
```
python src/train.py --predict-file sample_inputs.csv
```

## Outputs
- `outputs/metrics.json`: evaluation metrics and selected K
- `outputs/classification_report.json`: detailed class report
- `outputs/confusion_matrix.png`: confusion matrix plot
- `outputs/k_score.png`: CV score vs K (for KNN)
- `outputs/model.joblib`: trained model pipeline
- `outputs/predictions.csv`: predictions for provided input file (label id + name)

## What You Learn
- Data loading and inspection
- Train/test splitting with stratification
- Feature scaling
- Supervised learning with K-Nearest Neighbors
- Model selection using cross-validation
- Evaluation using accuracy, precision, recall, and F1

## Next Ideas
- Swap in another dataset (Wine, Breast Cancer)
- Add a new model (SVM, Random Forest)
- Try different scoring metrics for tuning (macro F1)
