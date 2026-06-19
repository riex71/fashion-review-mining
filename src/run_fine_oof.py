from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import KFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large_oof"

TRAIN_CSV = DATA_DIR / "train_fine.csv"

MODEL_NAME = "klue/roberta-large"
MAX_LENGTH = 256
SEED = 42
N_SPLITS = 5
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

    return {
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "subset_accuracy": float((preds == labels).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_df(TRAIN_CSV)
    print(f"rows: {len(df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_probs = np.zeros((len(df), NUM_LABELS), dtype=np.float32)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(df), start=1):
        print(f"\n===== FOLD {fold}/{N_SPLITS} =====")
        fold_dir = OUTPUT_DIR / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

        train_ds = to_dataset(train_df)
        valid_ds = to_dataset(valid_df)

        def tokenize_fn(batch):
            return tokenizer(
                batch["content_norm"],
                truncation=True,
                max_length=MAX_LENGTH,
            )

        train_ds = train_ds.map(tokenize_fn, batched=True)
        valid_ds = valid_ds.map(tokenize_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )

        training_args = TrainingArguments(
            output_dir=str(fold_dir),
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
            seed=SEED + fold,
            do_train=True,
            do_eval=True,
        )

        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))

        pred_output = trainer.predict(valid_ds)
        logits = pred_output.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()

        oof_probs[valid_idx] = probs

        fold_metric = {
            "fold": fold,
            **eval_metrics,
        }
        fold_rows.append(fold_metric)

    # OOF 전체 저장
    oof_df = df.copy()
    for i, label in enumerate(FINE_LABELS):
        oof_df[f"oof_prob__{label}"] = oof_probs[:, i]
        oof_df[f"oof_pred__{label}"] = (oof_probs[:, i] >= THRESHOLD).astype(int)

    y_true = oof_df[FINE_LABELS].astype(int).values
    y_pred = oof_df[[f"oof_pred__{label}" for label in FINE_LABELS]].astype(int).values

    oof_summary = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    # 부가 지표
    oof_df["gold_label_count"] = oof_df[FINE_LABELS].sum(axis=1)
    oof_df["oof_pred_label_count"] = oof_df[[f"oof_pred__{label}" for label in FINE_LABELS]].sum(axis=1)

    mean_conf = []
    max_conf = []
    for idx in range(len(oof_df)):
        probs_row = oof_probs[idx]
        mean_conf.append(float(np.mean(probs_row)))
        max_conf.append(float(np.max(probs_row)))
    oof_df["oof_mean_prob"] = mean_conf
    oof_df["oof_max_prob"] = max_conf

    folds_df = pd.DataFrame(fold_rows)

    oof_path = OUTPUT_DIR / "fine_oof_predictions.csv"
    folds_path = OUTPUT_DIR / "fold_metrics.csv"
    summary_path = OUTPUT_DIR / "oof_summary.json"

    oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
    folds_df.to_csv(folds_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(oof_summary, f, ensure_ascii=False, indent=2)

    print("\n===== OOF SUMMARY =====")
    print(json.dumps(oof_summary, indent=2, ensure_ascii=False))
    print(f"saved: {oof_path}")
    print(f"saved: {folds_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()