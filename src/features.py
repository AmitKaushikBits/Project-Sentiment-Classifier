"""
features.py. Fit a TF-IDF vectorizer on the valid rows only, and save it as a reusable artifact —
training and serving must use the exact same fitted vectorizer
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def run(clean_path: str, vectorizer_out: str, matrix_meta_out: str):
    df = pd.read_csv(clean_path)
    df = df[df["is_valid"] == True].copy()
    df["clean_text"] = df["clean_text"].fillna("")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])

    Path(vectorizer_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_out)

    meta = {
        "n_valid_rows": len(df),
        "vocab_size": len(vectorizer.vocabulary_),
        "ngram_range": vectorizer.ngram_range,
        "max_features": vectorizer.max_features,
    }
    with open(matrix_meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    return vectorizer, X, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/clean_reviews.csv")
    ap.add_argument("--vectorizer_out", default="models/tfidf_vectorizer.joblib")
    ap.add_argument("--meta_out", default="reports/feature_meta.json")
    args = ap.parse_args()
    run(args.clean, args.vectorizer_out, args.meta_out)
