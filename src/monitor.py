"""
Monitor production predictions and trigger retraining when drift is high.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_LABELS = ("negative", "neutral", "positive")

DEFAULT_THRESHOLDS = {
    "text_length_psi": 0.20,
    "label_psi": 0.25,
    "edge_rate": 0.10,
}

MIN_OBSERVATIONS = 1
MIN_RETRAIN_OBSERVATIONS = 500

PSI_THRESHOLDS = {
    "stable": 0.10,
    "watch": 0.25,
    "critical": 0.50,
}


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Prediction log not found: {path}")

    return pd.read_csv(path)


def _label_distribution(
    values,
    labels,
) -> list[float]:
    counts = (
        pd.Series(values).astype("string").value_counts().reindex(labels, fill_value=0)
    )

    total = max(int(counts.sum()), 1)

    return [float(count / total) for count in counts]


def population_stability_index(
    reference: list[float],
    current: list[float],
    epsilon: float = 1e-6,
) -> float:

    if len(reference) != len(current):
        raise ValueError("reference and current distributions must have equal length")

    psi = 0.0

    for expected, actual in zip(reference, current):
        expected = max(float(expected), epsilon)
        actual = max(float(actual), epsilon)

        psi += (actual - expected) * math.log(actual / expected)

    return float(psi)


def _length_distribution(
    values: pd.Series,
    bins: list[float],
) -> list[float]:
    lengths = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    bucketed = pd.cut(
        lengths,
        bins=bins,
        labels=False,
        include_lowest=True,
    )

    return _label_distribution(
        bucketed,
        tuple(range(len(bins) - 1)),
    )


def classify_psi(psi: float) -> str:

    if psi < PSI_THRESHOLDS["stable"]:
        return "stable"

    if psi < PSI_THRESHOLDS["watch"]:
        return "watch"

    if psi < PSI_THRESHOLDS["critical"]:
        return "drift"

    return "critical"


def determine_drift(
    observations: int,
    label_psi: float,
    text_length_psi: float,
    edge_rate: float,
    thresholds: dict[str, float],
) -> tuple[bool, str, int]:

    if observations < MIN_OBSERVATIONS:
        return False, "insufficient_data", 0

    score = 0

    if label_psi > thresholds["label_psi"]:
        score += 1

    if text_length_psi > thresholds["text_length_psi"]:
        score += 1

    if edge_rate > thresholds["edge_rate"]:
        score += 1

    if score == 0:
        return False, "stable", score

    if score == 1:
        return False, "watch", score

    if score == 2:
        return True, "high", score

    return True, "critical", score


def _build_summary(
    overall_status: str,
    drift_status: str,
    drift_detected: bool,
    observations: int,
    observations_source: str,
) -> dict[str, Any]:

    status_messages = {
        "healthy": "No material drift detected; model remains stable.",
        "monitor": "Minor drift detected; continue monitoring the production sample.",
        "action_required": "Drift exceeds the configured threshold; review the model and retrain if needed.",
        "insufficient_data": "Insufficient observations to assess drift reliably.",
    }

    return {
        "status": overall_status,
        "drift_status": drift_status,
        "drift_detected": drift_detected,
        "observations": int(observations),
        "source": str(observations_source),
        "message": status_messages.get(overall_status, "Drift status updated."),
    }


def build_report(
    reference_path: str | Path,
    prediction_log_path: str | Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:

    thresholds = {
        **DEFAULT_THRESHOLDS,
        **(thresholds or {}),
    }

    reference = pd.read_csv(reference_path)

    if reference.empty:
        raise ValueError("Reference dataset is empty")

    if "label" not in reference.columns:
        raise ValueError("Reference dataset must contain 'label'")

    if "is_valid" in reference.columns:
        reference = reference[reference["is_valid"].astype(bool)]

    predictions = _read_csv(prediction_log_path)

    if predictions.empty:
        raise ValueError("Prediction log contains no observations")

    labels = tuple(sorted(set(DEFAULT_LABELS) | set(reference["label"].dropna())))

    bins = [0, 5, 10, 20, 40, 80, float("inf")]

    reference_lengths = (
        reference.get(
            "clean_text",
            pd.Series(dtype=str),
        )
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    prediction_lengths = pd.to_numeric(
        predictions.get(
            "text_length",
            predictions.get("n_tokens"),
        ),
        errors="coerce",
    ).fillna(0)

    reference_label_dist = _label_distribution(
        reference["label"],
        labels,
    )

    prediction_labels = predictions.get(
        "label",
        predictions.get("predicted_label"),
    )

    if prediction_labels is None:
        raise ValueError("Prediction log must contain label or predicted_label")

    current_label_dist = _label_distribution(
        prediction_labels,
        labels,
    )

    text_length_psi = population_stability_index(
        _length_distribution(
            reference_lengths,
            bins,
        ),
        _length_distribution(
            prediction_lengths,
            bins,
        ),
    )

    label_psi = population_stability_index(
        reference_label_dist,
        current_label_dist,
    )

    if "is_edge_case" in predictions.columns:
        edge_rate = float(
            predictions["is_edge_case"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
            .mean()
        )
    else:
        edge_rate = 0.0

    drift_detected, drift_status, drift_score = determine_drift(
        observations=len(predictions),
        label_psi=label_psi,
        text_length_psi=text_length_psi,
        edge_rate=edge_rate,
        thresholds=thresholds,
    )

    if drift_status == "stable":
        overall_status = "healthy"
    elif drift_status in ("watch", "insufficient_data"):
        overall_status = "monitor"
    else:
        overall_status = "action_required"

    report = {
        "observations": int(len(predictions)),
        "reference_rows": int(len(reference)),
        "overall_status": overall_status,
        "drift_score": drift_score,
        "summary": _build_summary(
            overall_status=overall_status,
            drift_status=drift_status,
            drift_detected=drift_detected,
            observations=int(len(predictions)),
            observations_source=str(prediction_log_path),
        ),
        "label_distribution": {
            "reference": dict(zip(labels, reference_label_dist)),
            "current": dict(zip(labels, current_label_dist)),
        },
        "text_length_psi": round(
            text_length_psi,
            4,
        ),
        "label_psi": round(
            label_psi,
            4,
        ),
        "edge_rate": round(
            edge_rate,
            4,
        ),
        "thresholds": thresholds,
        "drift_detected": drift_detected,
        "drift_status": drift_status,
        "psi_status": {
            "text_length": classify_psi(text_length_psi),
            "label": classify_psi(label_psi),
        },
    }

    return report


def run(
    reference_path: str | Path = "data/clean_reviews.csv",
    prediction_log_path: str | Path = "logs/predictions_log.csv",
    report_path: str | Path = "reports/monitoring_report.json",
    retrain: bool = False,
    drift_csv: str | Path = "data/drift_reviews.csv",
    baseline: str | Path | None = "data/clean_reviews.csv",
) -> dict[str, Any]:

    baseline_path = baseline or reference_path

    drift_csv = Path(drift_csv)

    observations_path = drift_csv if drift_csv.exists() else prediction_log_path

    report = build_report(
        baseline_path,
        observations_path,
    )

    report["observations_source"] = str(observations_path)
    report["summary"] = _build_summary(
        overall_status=report["overall_status"],
        drift_status=report["drift_status"],
        drift_detected=report["drift_detected"],
        observations=report["observations"],
        observations_source=str(observations_path),
    )

    report["retraining"] = None

    should_retrain = (
        retrain
        and report["drift_detected"]
        and report["drift_status"] in {"high", "critical"}
        and report["observations"] >= MIN_RETRAIN_OBSERVATIONS
    )

    if should_retrain:

        from src.train import run as train_model

        report["retraining"] = train_model(
            str(baseline_path),
            "models/tfidf_vectorizer.joblib",
            "models",
            "reports/model_comparison.json",
        )

    report_path = Path(report_path)

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--reference",
        default="data/clean_reviews.csv",
    )

    parser.add_argument(
        "--baseline",
        default="data/clean_reviews.csv",
    )

    parser.add_argument(
        "--predictions",
        default="logs/predictions_log.csv",
    )

    parser.add_argument(
        "--drift_csv",
        default="reports/drift_reviews.csv",
    )

    parser.add_argument(
        "--report",
        default="reports/monitoring_report.json",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
    )

    args = parser.parse_args()

    report = run(
        reference_path=args.reference,
        prediction_log_path=args.predictions,
        report_path=args.report,
        drift_csv=args.drift_csv,
        baseline=args.baseline,
        retrain=args.retrain,
    )

    summary = report["summary"]

    print("\nDrift monitoring report")
    print("-" * 72)
    print(
        f"Status: {summary['status'].upper()} | "
        f"Detected: {summary['drift_detected']} | "
        f"Drift level: {summary['drift_status']} | "
        f"Observations: {summary['observations']}"
    )
    print(f"Source: {summary['source']}")
    print(f"Message: {summary['message']}")
    print(
        "Key metrics: "
        f"label_psi={report['label_psi']} | "
        f"text_length_psi={report['text_length_psi']} | "
        f"edge_rate={report['edge_rate']}"
    )
    print("-" * 72)
