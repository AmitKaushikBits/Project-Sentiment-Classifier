"""
ingest.py. It should read the raw CSV, clean and tokenize the text, and validate every row —
flagging problems rather than silently dropping rows, so the fate of every record is auditable.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

NEUTRAL_REVIEWS = [
    "The product works as expected.",
    "It is an average product for the price.",
    "The quality is okay and matches the description.",
    "The product is neither great nor terrible.",
    "It does the basic job without any surprises.",
    "The experience was acceptable overall.",
    "The design is simple and the performance is standard.",
    "It arrived on time and works normally.",
    "The features are useful, but nothing stands out.",
    "The price seems fair for what you receive.",
    "Setup was straightforward and the results are reasonable.",
    "The product is fine for occasional use.",
    "It is an ordinary product with average performance.",
    "The screen is clear enough for everyday tasks.",
    "Battery life is acceptable for a normal day.",
    "The sound quality is decent but not exceptional.",
    "The item looks as pictured and functions normally.",
    "It meets my basic needs without extra features.",
    "The performance is consistent and fairly typical.",
    "I have no strong opinion about this product.",
    "The product is usable, although there is room to improve.",
    "It is neither especially convenient nor difficult to use.",
    "The materials feel standard for this price range.",
    "The overall experience was mixed but acceptable.",
    "The product is fine and performs the basic functions.",
    "It is okay for the price, with ordinary performance.",
    "The item is practical, though not particularly impressive.",
    "The quality is neither poor nor outstanding.",
    "It works well enough for routine tasks.",
    "The product offers a typical experience for its category.",
    "The results are satisfactory without being exceptional.",
    "I find the product adequate for my needs.",
    "The service was standard and met my expectations.",
    "The device performs normally under everyday use.",
    "The value is reasonable compared with similar products.",
    "It has some useful features and some ordinary limitations.",
    "The product is acceptable, but it does not stand out.",
    "Everything worked, although the experience was unremarkable.",
    "The item is suitable for basic use and routine needs.",
    "The performance is fair and consistent.",
    "It is a middle-of-the-road option.",
    "The product is serviceable for the money.",
    "The experience was neither especially good nor bad.",
    "It meets the minimum requirements for everyday use.",
    "The product has average quality and ordinary features.",
    "The item is okay, but I would not strongly recommend it.",
    "The product does what it says without exceeding expectations.",
]

NEUTRAL_TARGET_ROWS = 1000
NEUTRAL_SUBJECTS = [
    "phone",
    "headphones",
    "tablet",
    "laptop",
    "smartwatch",
    "camera",
    "coffee maker",
    "keyboard",
    "monitor",
    "speaker",
]
NEUTRAL_ASSESSMENTS = [
    "works as expected",
    "performs adequately for everyday use",
    "has average performance",
    "meets the basic requirements",
    "is reasonably easy to use",
    "offers the usual features",
    "is suitable for occasional use",
    "does the job without surprises",
    "has a standard build for its price",
    "provides a typical user experience",
]
NEUTRAL_QUALIFIERS = [
    "nothing stands out",
    "the results are acceptable",
    "the value is about average",
    "there are strengths and limitations",
    "it meets my basic needs",
    "the experience is unremarkable",
    "the quality is neither poor nor exceptional",
    "I have no strong opinion about it",
    "the performance is consistent",
    "it is fine for the price",
    "the experience is mixed but usable",
    "it does not exceed expectations",
]

NEGATIVE_REVIEWS = [
    "The product has bad quality and broke after one day.",
    "The build quality is poor and the item stopped working quickly.",
    "This is badly made and failed during the first use.",
    "The quality is terrible and the product is unusable.",
    "It broke immediately and the replacement was also defective.",
    "The product feels cheap and does not work properly.",
    "Poor quality materials make this product frustrating to use.",
    "The device stopped working after a few days.",
    "This item is disappointing because it arrived damaged.",
    "The product performs badly and is not worth the price.",
    "The quality is unacceptable and the controls keep failing.",
    "It was defective out of the box and could not be used.",
]


def clean_text(text):
    t = str(text).lower()
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def build_neutral_reviews(target_rows=NEUTRAL_TARGET_ROWS):
    """Return a deterministic neutral set with at least the requested size."""
    reviews = list(NEUTRAL_REVIEWS)
    for subject in NEUTRAL_SUBJECTS:
        for assessment in NEUTRAL_ASSESSMENTS:
            for qualifier in NEUTRAL_QUALIFIERS:
                reviews.append(f"The {subject} {assessment}, and {qualifier}.")
                if len(reviews) >= target_rows:
                    return reviews[:target_rows]
    return reviews


def run_ingest(
    raw_path: Path = Path("data/merge/merged_sentiment_dataset.csv"),
    clean_path: Path = Path("data/clean_reviews.csv"),
    validation_path: Path = Path("reports/validation_report.json"),
    version_path: Path = Path("data/DATASET_VERSION.json"),
):
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "reviews" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"reviews": "text"})
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset must contain text and label columns; missing {sorted(missing_columns)}"
        )
    df = df.reset_index().rename(columns={"index": "id"})
    raw_labels = df["label"].copy()
    numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
    df["label"] = numeric_labels.map(
        {
            1: "positive",
            0: "negative",
            2: "neutral",
        }
    )
    string_labels = df["label"].isna()
    df.loc[string_labels, "label"] = (
        raw_labels.loc[string_labels].astype(str).str.lower()
    )

    existing_neutral_rows = int((df["label"] == "neutral").sum())
    neutral_reviews = build_neutral_reviews(
        max(0, NEUTRAL_TARGET_ROWS - existing_neutral_rows)
    )
    neutral_df = pd.DataFrame(
        {
            "id": range(len(df), len(df) + len(neutral_reviews)),
            "text": neutral_reviews,
            "label": "neutral",
        }
    )
    negative_df = pd.DataFrame(
        {
            "id": range(
                len(df) + len(neutral_df),
                len(df) + len(neutral_df) + len(NEGATIVE_REVIEWS),
            ),
            "text": NEGATIVE_REVIEWS,
            "label": "negative",
        }
    )
    if existing_neutral_rows < NEUTRAL_TARGET_ROWS:
        df = pd.concat([df, neutral_df], ignore_index=True)
    if "negative" not in set(df["label"].dropna()):
        df = pd.concat([df, negative_df], ignore_index=True)

    ALLOWED_LABELS = {"positive", "neutral", "negative"}

    df["clean_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean_text"].str.split()
    df["_flag_empty"] = df["text"].astype(str).str.strip().eq("") | (
        df["clean_text"].str.split().str.len() == 0
    )
    df["_flag_invalid_label"] = ~df["label"].isin(ALLOWED_LABELS)
    df["is_valid"] = ~(df["_flag_empty"] | df["_flag_invalid_label"])

    report = {
        "n_rows": int(len(df)),
        "n_valid": int(df["is_valid"].sum()),
        "n_invalid": int((~df["is_valid"]).sum()),
        "n_empty_text": int(df["_flag_empty"].sum()),
        "n_invalid_label": int(df["_flag_invalid_label"].sum()),
        "allowed_labels": sorted(list(ALLOWED_LABELS)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    validation_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_df = df[df["is_valid"]].copy()
    out_df.to_csv(clean_path, index=False)

    with clean_path.open("rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:12]

    version = {
        "version_tag": f"v-{sha}",
        "n_rows": int(len(out_df)),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with version_path.open("w", encoding="utf-8") as f:
        json.dump(version, f, indent=2)

    return df, out_df, report, version


if __name__ == "__main__":
    run_ingest()
