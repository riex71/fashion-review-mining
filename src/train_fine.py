from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, hamming_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large"

TRAIN_CSV = DATA_DIR / "train_fine.csv"
EVAL_CSV = DATA_DIR / "eval_fine.csv"

MODEL_NAME = "klue/roberta-large"
MAX_LENGTH = 256
SEED = 42
THRESHOLD = 0.5

FINE_LABELS = [
    "late_delivery",
    "wrong_item",
    "missing_item",
    "damaged_on_arrival",
    "packaging_issue",
    "size_large",
    "size_small",
    "fit_dissatisfaction",
    "finish_defect",
    "durability",
    "contamination",
    "fabric_issue",
    "thickness",
    "poor_comfort",
    "color_mismatch",
    "design_mismatch",
    "price_dissatisfaction",
    "other_complaint",
]

NUM_LABELS = len(FINE_LABELS)


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

    required = ["sample_id", "content_norm"] + FINE_LABELS
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path.name}에 필수 컬럼이 없습니다: {col}")

    df = df.copy()
    df["content_norm"] = df["content_norm"].fillna("").astype(str).str.strip()
    df = df[df["content_norm"] != ""].reset_index(drop=True)

    for col in FINE_LABELS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    labels = df[FINE_LABELS].values.tolist()
    ds_df = pd.DataFrame({
        "sample_id": df["sample_id"].tolist(),
        "content_norm": df["content_norm"].tolist(),
        "labels": labels,
    })
    return Dataset.from_pandas(ds_df, preserve_index=False)


class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        labels = labels.to(logits.device).float()
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    labels = np.array(labels).astype(int)

    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    subset_acc = float((preds == labels).all(axis=1).mean())
    ham_loss = hamming_loss(labels, preds)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "subset_accuracy": float(subset_acc),
        "hamming_loss": float(ham_loss),
    }


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_df(TRAIN_CSV)
    eval_df = load_df(EVAL_CSV)

    print(f"train rows: {len(train_df)}")
    print(f"eval rows: {len(eval_df)}")
    print("train label counts:")
    print(train_df[FINE_LABELS].sum().sort_values(ascending=False))
    print("\neval label counts:")
    print(eval_df[FINE_LABELS].sum().sort_values(ascending=False))

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
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
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
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        do_train=True,
        do_eval=True,
    )

    trainer = MultiLabelTrainer(
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
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)

    pred_df = eval_df.copy()
    for i, label in enumerate(FINE_LABELS):
        pred_df[f"pred_prob__{label}"] = probs[:, i]
        pred_df[f"pred__{label}"] = preds[:, i]

    pred_path = OUTPUT_DIR / "eval_predictions.csv"
    metrics_path = OUTPUT_DIR / "eval_metrics.json"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    print(f"saved predictions: {pred_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()