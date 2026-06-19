from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score, hamming_loss


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt" / "eval_predictions.csv"
OUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt"

OUT_LABEL_CSV = OUT_DIR / "label_metrics.csv"
OUT_SUMMARY_JSON = OUT_DIR / "label_metrics_summary.json"

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


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)

    required_gold = FINE_LABELS
    required_pred = [f"pred__{label}" for label in FINE_LABELS]

    for col in required_gold + required_pred:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    y_true = df[FINE_LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
    y_pred = df[[f"pred__{label}" for label in FINE_LABELS]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0).astype(int).values

    rows = []
    for i, label in enumerate(FINE_LABELS):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        precision, recall, f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            average="binary",
            zero_division=0,
        )

        support = int(yt.sum())

        rows.append({
            "label": label,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "pred_positive": int(yp.sum()),
            "gold_positive": support,
        })

    label_df = pd.DataFrame(rows).sort_values(by=["f1", "support"], ascending=[False, False])
    label_df.to_csv(OUT_LABEL_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(label_df.to_string(index=False))
    print("\nSUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsaved: {OUT_LABEL_CSV}")
    print(f"saved: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()