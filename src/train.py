import argparse
import json
import os
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    target_names: list
    feature_names: list


def load_dataset(name):
    if name == "iris":
        dataset = load_iris(as_frame=True)
    elif name == "wine":
        dataset = load_wine(as_frame=True)
    elif name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
    else:
        raise ValueError("Unsupported dataset.")

    return DatasetBundle(
        features=dataset.data,
        target=dataset.target,
        target_names=list(dataset.target_names),
        feature_names=list(dataset.feature_names),
    )


def split_data(features, target, test_size, random_state):
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_knn_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier()),
        ]
    )


def tune_knn(pipeline, x_train, y_train, scoring):
    param_grid = {
        "knn__n_neighbors": list(range(1, 31, 2)),
        "knn__weights": ["uniform", "distance"],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    return grid


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average=None
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(y_test, predictions, average="weighted")
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
        },
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
    }


def save_confusion_matrix(model, x_test, y_test, target_names, output_dir):
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()


def save_k_accuracy(grid, output_dir, scoring):
    results = grid.cv_results_
    k_values = results["param_knn__n_neighbors"].data
    mean_scores = results["mean_test_score"]

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, mean_scores, marker="o")
    plt.title(f"KNN Cross-Validation ({scoring})")
    plt.xlabel("K (n_neighbors)")
    plt.ylabel("Mean CV Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "k_score.png")
    plt.savefig(output_path)
    plt.close()


def compare_models(x_train, y_train, x_test, y_test):
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=200),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(random_state=0),
    }
    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[name] = metrics

    return results


def ensure_output_dir():
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_prediction_input(path, feature_names):
    if path.lower().endswith(".csv"):
        data = pd.read_csv(path)
    elif path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        data = pd.DataFrame(payload)
    else:
        raise ValueError("Prediction input must be .csv or .json")

    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise ValueError(
            "Prediction file missing required columns: " + ", ".join(missing)
        )
    return data[feature_names]


def run_predictions(model_path, input_path, feature_names, output_dir):
    model = joblib.load(model_path)
    input_data = load_prediction_input(input_path, feature_names)
    predictions = model.predict(input_data)

    label_names = getattr(model, "classes_", None)
    if label_names is not None:
        labels = [str(label_names[label]) for label in predictions]
    else:
        labels = [str(label) for label in predictions]

    output = pd.DataFrame({"prediction_id": predictions, "prediction": labels})
    output_path = os.path.join(output_dir, "predictions.csv")
    output.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a KNN classifier on a built-in dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["iris", "wine", "breast_cancer"],
        default="iris",
        help="Dataset to use for training.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Evaluate extra baseline models.",
    )
    parser.add_argument(
        "--tune-metric",
        choices=["accuracy", "f1_macro"],
        default="accuracy",
        help="Metric to tune KNN hyperparameters.",
    )
    parser.add_argument(
        "--predict-file",
        help="Optional .csv or .json with feature columns to predict.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test = split_data(
        dataset.features,
        dataset.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output_dir = ensure_output_dir()

    knn_pipeline = build_knn_pipeline()
    grid = tune_knn(knn_pipeline, x_train, y_train, args.tune_metric)

    best_model = grid.best_estimator_
    metrics = evaluate_model(best_model, x_test, y_test)

    save_confusion_matrix(
        best_model, x_test, y_test, dataset.target_names, output_dir
    )
    save_k_accuracy(grid, output_dir, args.tune_metric)

    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(best_model, model_path)

    results = {
        "best_params": grid.best_params_,
        "dataset": args.dataset,
        "tune_metric": args.tune_metric,
        "metrics": metrics,
    }

    if args.compare_models:
        results["baseline_models"] = compare_models(
            x_train, y_train, x_test, y_test
        )

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    report_path = os.path.join(output_dir, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(metrics["classification_report"], handle, indent=2)

    if args.predict_file:
        prediction_path = run_predictions(
            model_path, args.predict_file, dataset.feature_names, output_dir
        )
        print(f"Predictions saved to: {prediction_path}")

    print("Training complete.")
    print(f"Best params: {grid.best_params_}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
