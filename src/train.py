import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

EXPERIMENT_NAME = "ML Mini Project - Sentiment Classifier"


def load_data(clean_path, vectorizer_path):
    df = pd.read_csv(clean_path)

    df = df[df["is_valid"] == True].copy()
    df["clean_text"] = df["clean_text"].fillna("")

    vectorizer = joblib.load(vectorizer_path)

    X = vectorizer.transform(df["clean_text"])
    y = df["label"]

    return X, y, vectorizer, df


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "precision_macro": precision_score(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        ),
    }

    return metrics, preds


def print_confusion_matrix(matrix, labels):
    matrix_frame = pd.DataFrame(
        matrix,
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(matrix_frame.to_string())


def print_model_summary(name, metrics):
    print(f"\n{name.upper()} SUMMARY")
    for metric_name in (
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
    ):
        display_name = metric_name.replace("_", " ").title()
        print(f"  {display_name:<18} {metrics[metric_name]:.4f}")


def run(clean_path, vectorizer_path, model_dir, report_out, seed=42):

    X, y, vectorizer, df = load_data(
        clean_path,
        vectorizer_path,
    )

    print("\nClass Distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            C=2.0,
            random_state=seed,
            max_iter=10000,
        ),
    }

    Path(model_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {}

    for name, model in candidates.items():

        with mlflow.start_run(run_name=name):

            mlflow.log_param("model_type", name)
            mlflow.log_param("train_rows", X_train.shape[0])
            mlflow.log_param("test_rows", X_test.shape[0])

            for k, v in model.get_params().items():
                mlflow.log_param(f"hp_{k}", str(v))

            model.fit(X_train, y_train)

            metrics, preds = evaluate(
                model,
                X_test,
                y_test,
            )

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            report_dict = classification_report(
                y_test,
                preds,
                output_dict=True,
                zero_division=0,
            )

            report_txt = classification_report(
                y_test,
                preds,
                zero_division=0,
            )

            print(f"\n{name.upper()} REPORT")
            print(report_txt)

            for label in y.unique():

                label = str(label)

                if label in report_dict:
                    mlflow.log_metric(
                        f"f1_{label}",
                        report_dict[label]["f1-score"],
                    )

                    mlflow.log_metric(
                        f"precision_{label}",
                        report_dict[label]["precision"],
                    )

                    mlflow.log_metric(
                        f"recall_{label}",
                        report_dict[label]["recall"],
                    )

            cm = confusion_matrix(
                y_test,
                preds,
                labels=sorted(y.unique()),
            ).tolist()

            report_path = f"reports/{name}_classification_report.txt"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_txt)

            mlflow.log_artifact(report_path)

            model_path = f"{model_dir}/{name}.joblib"
            joblib.dump(model, model_path)

            mlflow.sklearn.log_model(
                model,
                artifact_path=name,
            )

            results[name] = {
                **metrics,
                "confusion_matrix": cm,
                "labels": sorted(y.unique().tolist()),
            }

            print_confusion_matrix(cm, results[name]["labels"])
            print_model_summary(name, metrics)

    best_name = max(results, key=lambda x: results[x]["f1_macro"])

    justification = (
        f"{best_name} selected because it achieved "
        f"highest Macro-F1 score "
        f"({results[best_name]['f1_macro']:.4f})."
    )

    summary = {
        "results": results,
        "best_model": best_name,
        "justification": justification,
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    dvc_metrics = {
        name: {
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
        }
        for name, r in results.items()
    }

    dvc_metrics["best_model"] = best_name

    with open(
        "reports/metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            dvc_metrics,
            f,
            indent=2,
        )

    best_src = f"{model_dir}/{best_name}.joblib"
    best_dst = f"{model_dir}/best_model.joblib"

    joblib.dump(
        joblib.load(best_src),
        best_dst,
    )

    with open(
        f"{model_dir}/best_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model_name": best_name,
                "metrics": results[best_name],
            },
            f,
            indent=2,
        )

    print("\nFINAL SUMMARY")
    for name, model_metrics in results.items():
        print_model_summary(name, model_metrics)
    print(f"\nSelected model: {best_name}")
    print(f"Reason: {justification}")

    return summary


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--clean",
        default="data/clean_reviews.csv",
    )

    ap.add_argument(
        "--vectorizer",
        default="models/tfidf_vectorizer.joblib",
    )

    ap.add_argument(
        "--model_dir",
        default="models",
    )

    ap.add_argument(
        "--report_out",
        default="reports/model_comparison.json",
    )

    args = ap.parse_args()

    run(
        args.clean,
        args.vectorizer,
        args.model_dir,
        args.report_out,
    )
