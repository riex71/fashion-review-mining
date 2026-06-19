from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "external_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 최종 채택 모델
MODEL_DIR = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt"
BASE_MODEL_NAME = "klue/roberta-large"
THRESHOLD_CSV = MODEL_DIR / "fine_labelwise_best_thresholds.csv"

# 외부 데이터 경로
ABLY_CSV = Path("./ably_reviews.csv")
ZIGZAG_CSV = Path("./zigzag_reviews.csv")

TEXT_COL = "content_norm"
MAX_LENGTH = 256
SEED = 42

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

FINE_TO_COARSE = {
    "delivery": ["late_delivery", "wrong_item", "missing_item", "damaged_on_arrival"],
    "packaging": ["packaging_issue"],
    "size_fit": ["size_large", "size_small", "fit_dissatisfaction"],
    "quality": ["finish_defect", "durability", "contamination"],
    "fabric_comfort": ["fabric_issue", "thickness", "poor_comfort"],
    "color_design": ["color_mismatch", "design_mismatch"],
    "price": ["price_dissatisfaction"],
    "other_complaint": ["other_complaint"],
}
COARSE_LABELS = list(FINE_TO_COARSE.keys())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"모델 폴더가 없습니다: {model_dir}")

    checkpoints = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )

    if not checkpoints:
        raise FileNotFoundError(f"checkpoint-* 폴더를 찾을 수 없습니다: {model_dir}")

    return checkpoints[-1]


def load_threshold_map(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"threshold csv가 없습니다: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "label" not in df.columns or "best_threshold" not in df.columns:
        raise ValueError("threshold csv에 label / best_threshold 컬럼이 없습니다.")
    return {str(r["label"]): float(r["best_threshold"]) for _, r in df.iterrows()}


def load_external_df(path: Path, platform_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{platform_name} 파일이 없습니다: {path.resolve()}")

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    if TEXT_COL not in df.columns:
        raise ValueError(f"{platform_name}에 {TEXT_COL} 컬럼이 없습니다.")

    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].reset_index(drop=True)

    if "review_id" not in df.columns:
        df["review_id"] = [f"{platform_name}_{i:08d}" for i in range(len(df))]
    if "goods_no" not in df.columns:
        df["goods_no"] = ""
    if "goods_name" not in df.columns:
        df["goods_name"] = ""
    if "analysis_category" not in df.columns:
        df["analysis_category"] = ""
    if "option_text" not in df.columns:
        df["option_text"] = ""

    df["platform"] = platform_name
    return df


def to_dataset(df: pd.DataFrame) -> Dataset:
    ds_df = pd.DataFrame({
        "row_id": np.arange(len(df)),
        "content_norm": df[TEXT_COL].tolist(),
    })
    return Dataset.from_pandas(ds_df, preserve_index=False)


def main():
    set_seed(SEED)

    threshold_map = load_threshold_map(THRESHOLD_CSV)
    checkpoint_dir = find_latest_checkpoint(MODEL_DIR)
    print(f"using model checkpoint: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint_dir),
        num_labels=len(FINE_LABELS),
        problem_type="multi_label_classification",
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    platform_configs = [
        ("ably", ABLY_CSV),
        ("zigzag", ZIGZAG_CSV),
    ]

    summary_rows = []
    coarse_rows = []
    fine_rows = []

    for platform_name, csv_path in platform_configs:
        df = load_external_df(csv_path, platform_name)
        print(f"{platform_name}: {len(df)} rows")

        ds = to_dataset(df)

        def tokenize_fn(batch):
            return tokenizer(
                batch["content_norm"],
                truncation=True,
                max_length=MAX_LENGTH,
            )

        ds = ds.map(tokenize_fn, batched=True)

        pred_output = trainer.predict(ds)
        logits = pred_output.predictions
        probs = torch.sigmoid(torch.tensor(logits)).numpy()

        out_df = df.copy()

        for i, label in enumerate(FINE_LABELS):
            prob_col = f"pred_prob__{label}"
            pred_col = f"pred_labelwise__{label}"
            th = threshold_map[label]

            out_df[prob_col] = probs[:, i]
            out_df[pred_col] = (out_df[prob_col] >= th).astype(int)

        for coarse_label, fine_labels in FINE_TO_COARSE.items():
            pred_cols = [f"pred_labelwise__{label}" for label in fine_labels]
            out_df[f"pred__{coarse_label}"] = out_df[pred_cols].max(axis=1)

        out_df["pred_complaint"] = out_df[[f"pred_labelwise__{label}" for label in FINE_LABELS]].max(axis=1)
        out_df["pred_fine_label_count"] = out_df[[f"pred_labelwise__{label}" for label in FINE_LABELS]].sum(axis=1)
        out_df["pred_coarse_label_count"] = out_df[[f"pred__{label}" for label in COARSE_LABELS]].sum(axis=1)
        out_df["max_fine_prob"] = out_df[[f"pred_prob__{label}" for label in FINE_LABELS]].max(axis=1)

        pred_path = OUTPUT_DIR / f"{platform_name}_predictions.csv"
        out_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        summary_rows.append({
            "platform": platform_name,
            "n_reviews": int(len(out_df)),
            "pred_complaint_rate": float(out_df["pred_complaint"].mean()),
            "avg_pred_fine_label_count": float(out_df["pred_fine_label_count"].mean()),
            "avg_pred_coarse_label_count": float(out_df["pred_coarse_label_count"].mean()),
            "avg_max_fine_prob": float(out_df["max_fine_prob"].mean()),
        })

        for label in COARSE_LABELS:
            coarse_rows.append({
                "platform": platform_name,
                "label": label,
                "pred_count": int(out_df[f"pred__{label}"].sum()),
                "pred_rate": float(out_df[f"pred__{label}"].mean()),
            })

        for label in FINE_LABELS:
            fine_rows.append({
                "platform": platform_name,
                "label": label,
                "pred_count": int(out_df[f"pred_labelwise__{label}"].sum()),
                "pred_rate": float(out_df[f"pred_labelwise__{label}"].mean()),
            })

        print(f"saved: {pred_path}")

    summary_df = pd.DataFrame(summary_rows)
    coarse_df = pd.DataFrame(coarse_rows).sort_values(["platform", "pred_rate"], ascending=[True, False])
    fine_df = pd.DataFrame(fine_rows).sort_values(["platform", "pred_rate"], ascending=[True, False])

    summary_df.to_csv(OUTPUT_DIR / "external_summary.csv", index=False, encoding="utf-8-sig")
    coarse_df.to_csv(OUTPUT_DIR / "external_coarse8_distribution.csv", index=False, encoding="utf-8-sig")
    fine_df.to_csv(OUTPUT_DIR / "external_fine_distribution.csv", index=False, encoding="utf-8-sig")

    print("\nSUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\nsaved: {OUTPUT_DIR / 'external_summary.csv'}")
    print(f"saved: {OUTPUT_DIR / 'external_coarse8_distribution.csv'}")
    print(f"saved: {OUTPUT_DIR / 'external_fine_distribution.csv'}")


if __name__ == "__main__":
    main()