import argparse
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch

EXPERIMENT_NAME = "ticket-sentiment-classifier"
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def main(
    clean_path="data/clean_reviews.csv",
    model_name="distilbert-base-uncased",
    epochs=3,
    seed=42,
):
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    from torch.utils.data import Dataset

    df = pd.read_csv(clean_path)
    df = df[df["is_valid"] == True].copy()
    df["y"] = df["label"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDS(Dataset):
        def __init__(self, frame):
            self.texts = frame["clean_text"].fillna("").tolist()
            self.labels = frame["y"].tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS)
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }

    args = TrainingArguments(
        output_dir="models/distilbert_sentiment",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TextDS(train_df),
        eval_dataset=TextDS(test_df),
        compute_metrics=compute_metrics,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="distilbert_finetuned"):
        mlflow.log_param("model_type", "distilbert_finetuned")
        mlflow.log_param("base_model", model_name)
        mlflow.log_param("epochs", epochs)
        trainer.train()
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k.replace("eval_", ""), v)
        trainer.save_model("models/distilbert_sentiment")
        mlflow.log_artifacts(
            "models/distilbert_sentiment", artifact_path="distilbert_model"
        )
        print(metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="data/clean_reviews.csv")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    main(clean_path=args.clean, epochs=args.epochs)
