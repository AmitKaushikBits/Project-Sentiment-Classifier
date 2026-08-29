"""Download, normalize, merge, and shuffle the project sentiment datasets."""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from shutil import which

import pandas as pd

if __package__:
    from .ingest import build_neutral_reviews
else:
    from ingest import build_neutral_reviews

DATASETS = ("kazanova/sentiment140", "marklvl/sentiment-labelled-sentences-data-set")
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
NEUTRAL_LABEL = 2
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
MERGE_DIR = BASE_DIR / "data" / "merge"
MODEL_DATA_DIR = BASE_DIR / "data" / "modelData"


def kaggle_command() -> list[str]:
    executable = which("kaggle")
    if executable:
        return [executable]
    if importlib.util.find_spec("kaggle") is not None:
        return [sys.executable, "-m", "kaggle"]
    raise RuntimeError(
        "Kaggle is not installed. Install it with `pip install kaggle` "
        "and configure Kaggle credentials."
    )


def download_dataset(dataset: str) -> None:
    command = kaggle_command() + [
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(RAW_DIR),
    ]
    print(f"Downloading {dataset}...")
    subprocess.run(command, check=True)


def extract_archives() -> None:
    for archive_path in RAW_DIR.glob("*.zip"):
        print(f"Extracting {archive_path.name}...")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(RAW_DIR)
        archive_path.unlink()


def load_legacy_dataset() -> pd.DataFrame | None:
    legacy_dir = RAW_DIR / "sentiment labelled sentences"
    paths = [
        legacy_dir / "amazon_cells_labelled.txt",
        legacy_dir / "imdb_labelled.txt",
        legacy_dir / "yelp_labelled.txt",
    ]
    if not all(path.exists() for path in paths):
        return None
    frame = pd.concat(
        [pd.read_csv(path, names=["sentence", "label"], sep="\t") for path in paths],
        ignore_index=True,
    )
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce").map(
        {0: NEGATIVE_LABEL, 1: POSITIVE_LABEL}
    )
    return frame


def load_sentiment140() -> pd.DataFrame | None:
    paths = list(RAW_DIR.glob("*.csv"))
    if not paths:
        return None
    source_path = max(paths, key=lambda path: path.stat().st_size)
    frame = pd.read_csv(
        source_path,
        names=["label", "id", "date", "query", "user", "sentence"],
        encoding="latin-1",
    )
    frame = frame[frame["label"].isin([0, 4])][["sentence", "label"]].copy()
    frame["label"] = frame["label"].map({0: NEGATIVE_LABEL, 4: POSITIVE_LABEL})
    per_class = min(5000, int(frame["label"].value_counts().min()))
    return (
        frame.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=42)
        .reset_index(drop=True)
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        download_dataset(dataset)
    extract_archives()

    frames = [
        frame
        for frame in (load_legacy_dataset(), load_sentiment140())
        if frame is not None
    ]
    if not frames:
        raise FileNotFoundError(f"No supported dataset found in {RAW_DIR}")

    merged = pd.concat(frames, ignore_index=True).dropna(subset=["sentence", "label"])
    merged["sentence"] = merged["sentence"].astype(str).str.strip()
    merged = merged[merged["sentence"].ne("")].reset_index(drop=True)
    neutral_frame = pd.DataFrame(
        {"sentence": build_neutral_reviews(), "label": NEUTRAL_LABEL}
    )
    merged = pd.concat([merged, neutral_frame], ignore_index=True)

    MERGE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGE_DIR / "merged_sentiment_dataset.csv", index=False)

    shuffled = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled.rename(columns={"sentence": "reviews"}, inplace=True)
    shuffled.to_csv(MODEL_DATA_DIR / "shuffled_sentiment_dataset.csv", index=False)
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Shuffled dataset shape: {shuffled.shape}")


if __name__ == "__main__":
    main()
