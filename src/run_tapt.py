from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "tapt_roberta_large"

INPUT_CSV = DATA_DIR / "musinsa_train_ready.csv"

MODEL_NAME = "klue/roberta-large"
TEXT_COL = "content_norm"   # 필요하면 여기만 수정
MAX_LENGTH = 256
SEED = 42

# 먼저 가볍게
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# 검증용으로 일부 분리
VALID_SIZE = 2000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts(path: Path, text_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if text_col not in df.columns:
        raise ValueError(f"{path.name}에 텍스트 컬럼이 없습니다: {text_col}")

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df = df[df[text_col] != ""].reset_index(drop=True)

    return df[[text_col]].copy()


def to_dataset(df: pd.DataFrame, text_col: str) -> Dataset:
    return Dataset.from_pandas(df[[text_col]], preserve_index=False)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_texts(INPUT_CSV, TEXT_COL)
    print(f"all rows: {len(df)}")

    # 검증용 일부 분리
    if len(df) <= VALID_SIZE:
        raise ValueError("데이터가 너무 적어서 valid 분리가 어렵습니다.")

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    valid_df = df.iloc[:VALID_SIZE].reset_index(drop=True)
    train_df = df.iloc[VALID_SIZE:].reset_index(drop=True)

    print(f"train rows: {len(train_df)}")
    print(f"valid rows: {len(valid_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = to_dataset(train_df, TEXT_COL)
    valid_ds = to_dataset(valid_df, TEXT_COL)

    def tokenize_fn(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
        )

    train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[TEXT_COL],
    )
    valid_ds = valid_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[TEXT_COL],
    )

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=200,
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
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print("TAPT EVAL METRICS")
    print(json.dumps(eval_metrics, indent=2, ensure_ascii=False))

    metrics_path = OUTPUT_DIR / "tapt_eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    print(f"saved: {metrics_path}")
    print(f"best checkpoint dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()