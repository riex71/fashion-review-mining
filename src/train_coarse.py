from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "coarse_roberta_large"

TRAIN_CSV = DATA_DIR / "train_coarse.csv"
EVAL_CSV = DATA_DIR / "eval_coarse.csv"

MODEL_NAME = "klue/roberta-large"
MAX_LENGTH = 256
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    required = ["sample_id", "content_norm", "complaint"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path.name}에 필수 컬럼이 없습니다: {col}")

    df = df.copy()
    df["content_norm"] = df["content_norm"].fillna("").astype(str).str.strip()
    df["complaint"] = pd.to_numeric(df["complaint"], errors="coerce").fillna(0).astype(int)

    df = df[df["content_norm"] != ""].reset_index(drop=True)
    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    ds_df = df[["sample_id", "content_norm", "complaint"]].rename(
        columns={"complaint": "labels"}
    )
    return Dataset.from_pandas(ds_df, preserve_index=False)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_df(TRAIN_CSV)
    eval_df = load_df(EVAL_CSV)

    print(f"train rows: {len(train_df)}")
    print(f"eval rows: {len(eval_df)}")
    print(f"train complaint rate: {train_df['complaint'].mean():.4f}")
    print(f"eval complaint rate: {eval_df['complaint'].mean():.4f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = to_dataset(train_df)
    eval_ds = to_dataset(eval_df)

    def tokenize_fn(batch):
        return tokenizer(
            batch["content_norm"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    eval_ds = eval_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print("EVAL METRICS")
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))

    pred_output = trainer.predict(eval_ds)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    pred_df = eval_df.copy()
    pred_df["pred_prob"] = probs
    pred_df["pred_label"] = preds

    pred_path = OUTPUT_DIR / "eval_predictions.csv"
    metrics_path = OUTPUT_DIR / "eval_metrics.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    print(f"saved predictions: {pred_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()