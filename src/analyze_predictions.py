import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor import build_report

LABEL_ORDER = ["negative", "neutral", "positive"]
COLOR_MAP = {
    "negative": "#ef4444",
    "neutral": "#f59e0b",
    "positive": "#22c55e",
}


def _normalise_label(value):
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def load_predictions(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction log not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Prediction log is empty: {csv_path}")

    label_column = "predicted_label" if "predicted_label" in df.columns else "label"
    if label_column not in df.columns:
        raise ValueError(
            "Prediction log must contain a label column named 'predicted_label' or 'label'"
        )

    df = df.copy()
    df["predicted_label"] = df[label_column].map(_normalise_label)

    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "n_tokens" in df.columns:
        df["n_tokens"] = pd.to_numeric(df["n_tokens"], errors="coerce").fillna(0)

    if "is_edge_case" in df.columns:
        df["is_edge_case"] = (
            df["is_edge_case"].astype(str).str.lower().isin({"true", "1", "yes"})
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df


def analyse_predictions(df: pd.DataFrame) -> dict:
    total = int(len(df))

    label_counts = (
        df["predicted_label"]
        .value_counts()
        .reindex(LABEL_ORDER + ["unknown"], fill_value=0)
    )

    label_distribution = {
        key: round(float(value / total), 4) if total else 0.0
        for key, value in label_counts.items()
        if key in LABEL_ORDER + ["unknown"]
    }

    confidence_values = df.get("confidence")
    confidence_summary = {}
    if confidence_values is not None:
        valid_confidence = confidence_values.dropna()
        if not valid_confidence.empty:
            confidence_summary = {
                "mean": round(float(valid_confidence.mean()), 4),
                "median": round(float(valid_confidence.median()), 4),
                "min": round(float(valid_confidence.min()), 4),
                "max": round(float(valid_confidence.max()), 4),
            }

    token_counts = df.get("n_tokens")
    token_summary = {}
    if token_counts is not None:
        valid_tokens = pd.to_numeric(token_counts, errors="coerce").dropna()
        if not valid_tokens.empty:
            token_summary = {
                "mean_tokens": round(float(valid_tokens.mean()), 2),
                "median_tokens": round(float(valid_tokens.median()), 2),
                "max_tokens": int(valid_tokens.max()),
            }

    edge_rate = None
    if "is_edge_case" in df.columns:
        edge_rate = round(float(df["is_edge_case"].mean()), 4)

    hourly_volume = None
    if "timestamp" in df.columns:
        valid_timestamps = df["timestamp"].dropna()
        if not valid_timestamps.empty:
            hourly_volume = {
                ts.isoformat(): int(count)
                for ts, count in (
                    valid_timestamps.dt.floor("h").value_counts().sort_index().items()
                )
            }

    return {
        "total_predictions": total,
        "label_counts": {key: int(label_counts.get(key, 0)) for key in LABEL_ORDER},
        "label_distribution": label_distribution,
        "confidence": confidence_summary,
        "token_stats": token_summary,
        "edge_case_rate": edge_rate,
        "hourly_volume": hourly_volume,
    }


def plot_predictions(df: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Prediction Log Analysis", fontsize=16, fontweight="bold")

    label_counts = (
        df["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    )
    axes[0, 0].bar(
        label_counts.index,
        label_counts.values,
        color=[COLOR_MAP.get(label, "#6b7280") for label in label_counts.index],
    )
    axes[0, 0].set_title("Sentiment Distribution")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=0)

    if "confidence" in df.columns:
        confidence_values = df["confidence"].dropna()
        if not confidence_values.empty:
            axes[0, 1].hist(
                confidence_values, bins=20, color="#3b82f6", edgecolor="black"
            )
            axes[0, 1].set_title("Confidence Distribution")
            axes[0, 1].set_xlabel("Confidence")
            axes[0, 1].set_ylabel("Frequency")
        else:
            axes[0, 1].text(0.5, 0.5, "No confidence data", ha="center", va="center")
            axes[0, 1].set_axis_off()
    else:
        axes[0, 1].text(0.5, 0.5, "No confidence data", ha="center", va="center")
        axes[0, 1].set_axis_off()

    if "timestamp" in df.columns and not df["timestamp"].dropna().empty:
        hourly = df["timestamp"].dropna().dt.floor("h").value_counts().sort_index()
        axes[1, 0].plot(hourly.index, hourly.values, color="#8b5cf6", marker="o")
        axes[1, 0].set_title("Prediction Volume Over Time")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Predictions/hour")
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=30)
    else:
        axes[1, 0].text(0.5, 0.5, "No timestamp data", ha="center", va="center")
        axes[1, 0].set_axis_off()

    if "is_edge_case" in df.columns:
        edge_counts = df["is_edge_case"].fillna(False).value_counts()
        labels = ["Normal", "Edge case"]
        values = [int(edge_counts.get(False, 0)), int(edge_counts.get(True, 0))]
        axes[1, 1].pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#10b981", "#f97316"],
        )
        axes[1, 1].set_title("Edge Case Rate")
    else:
        axes[1, 1].text(0.5, 0.5, "No edge-case data", ha="center", va="center")
        axes[1, 1].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_console_summary(summary: dict, monitor_report: dict | None = None):
    label_counts = summary.get("label_counts", {})
    label_distribution = summary.get("label_distribution", {})
    confidence = summary.get("confidence", {})
    token_stats = summary.get("token_stats", {})
    edge_rate = summary.get("edge_case_rate")

    print("\n=== Prediction summary ===")
    print(f"Total predictions: {summary.get('total_predictions', 0)}")
    print(
        "Sentiment distribution: "
        + ", ".join(f"{label}={label_counts.get(label, 0)}" for label in LABEL_ORDER)
    )
    print(
        "Share by label: "
        + ", ".join(
            f"{label}={label_distribution.get(label, 0.0):.2%}" for label in LABEL_ORDER
        )
    )
    print(
        "Confidence: "
        f"mean={confidence.get('mean', 0.0)} | "
        f"median={confidence.get('median', 0.0)} | "
        f"min={confidence.get('min', 0.0)} | "
        f"max={confidence.get('max', 0.0)}"
    )
    print(
        "Token stats: "
        f"mean={token_stats.get('mean_tokens', 0.0)} | "
        f"median={token_stats.get('median_tokens', 0.0)} | "
        f"max={token_stats.get('max_tokens', 0)}"
    )
    print(f"Edge-case rate: {edge_rate}")

    if monitor_report is not None:
        print("\n=== Drift monitoring summary ===")
        print(
            "Monitoring status: "
            f"overall={monitor_report.get('overall_status', 'unknown')} | "
            f"drift={monitor_report.get('drift_detected', False)} | "
            f"level={monitor_report.get('drift_status', 'unknown')}"
        )
        print(
            "Drift metrics: "
            f"label_psi={monitor_report.get('label_psi', 0.0)} | "
            f"text_length_psi={monitor_report.get('text_length_psi', 0.0)} | "
            f"edge_rate={monitor_report.get('edge_rate', 0.0)}"
        )


def plot_monitoring(report: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        "label_psi": report.get("label_psi", 0.0),
        "text_length_psi": report.get("text_length_psi", 0.0),
        "edge_rate": report.get("edge_rate", 0.0),
    }

    thresholds = report.get("thresholds", {})
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = list(metrics.keys())
    values = list(metrics.values())
    positions = range(len(labels))

    ax.bar(positions, values, color=["#60a5fa", "#a78bfa", "#f59e0b"])

    for idx, key in enumerate(labels):
        threshold = thresholds.get(key, 0.0)
        ax.axhline(
            threshold,
            color="#ef4444",
            linestyle="--",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.text(
            idx,
            max(values[idx], threshold) + 0.02,
            f"threshold={threshold}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Monitoring Drift Metrics")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Value")
    ax.set_ylim(0, max(max(values), max(thresholds.values(), default=0.0)) * 1.5 + 0.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    status = report.get("drift_status", "stable")
    ax.text(
        0.02,
        0.98,
        f"Status: {status.upper()} | Detected: {report.get('drift_detected', False)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "alpha": 0.9},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse and visualize a sentiment prediction log and optional monitoring drift report."
    )
    parser.add_argument(
        "--log",
        default="logs/predictions_log.csv",
        help="Path to the prediction CSV log.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference dataset for monitoring drift analysis.",
    )
    parser.add_argument(
        "--output",
        default="reports/prediction_analysis.png",
        help="Output path for the prediction analysis chart.",
    )
    parser.add_argument(
        "--monitor-output",
        default="reports/monitoring_analysis.png",
        help="Output path for the monitoring drift chart.",
    )
    parser.add_argument(
        "--summary-json",
        default="reports/prediction_analysis_summary.json",
        help="Output path for the JSON analysis summary.",
    )
    args = parser.parse_args()

    df = load_predictions(args.log)
    summary = analyse_predictions(df)

    output_path = Path(args.output)
    plot_predictions(df, output_path)

    monitor_report = None
    if args.reference:
        monitor_report = build_report(args.reference, args.log)
        monitor_output = Path(args.monitor_output)
        plot_monitoring(monitor_report, monitor_output)
        summary["monitoring"] = {
            "overall_status": monitor_report["overall_status"],
            "drift_detected": monitor_report["drift_detected"],
            "drift_status": monitor_report["drift_status"],
            "label_psi": monitor_report["label_psi"],
            "text_length_psi": monitor_report["text_length_psi"],
            "edge_rate": monitor_report["edge_rate"],
        }
        print(f"Monitoring chart saved to: {monitor_output}")

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_console_summary(summary, monitor_report)
    print(f"\nPrediction chart saved to: {output_path}")
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()
