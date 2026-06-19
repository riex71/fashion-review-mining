from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large"

PRED_CSV = OUT_DIR / "eval_predictions.csv"
THRESHOLD_CSV = OUT_DIR / "fine_labelwise_best_thresholds.csv"

OUT_PRED_CSV = OUT_DIR / "eval_predictions_labelwise_threshold.csv"
OUT_LABEL_CSV = OUT_DIR / "label_metrics_labelwise_threshold.csv"
OUT_SUMMARY_JSON = OUT_DIR / "labelwise_threshold_summary.json"

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
    if not THRESHOLD_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {THRESHOLD_CSV}")

    pred_df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)
    th_df = pd.read_csv(THRESHOLD_CSV, encoding="utf-8-sig", low_memory=False)

    for label in FINE_LABELS:
        if label not in pred_df.columns:
            raise ValueError(f"gold 컬럼이 없습니다: {label}")
        if f"pred_prob__{label}" not in pred_df.columns:
            raise ValueError(f"확률 컬럼이 없습니다: pred_prob__{label}")

    if "label" not in th_df.columns or "best_threshold" not in th_df.columns:
        raise ValueError("fine_labelwise_best_thresholds.csv에 label / best_threshold 컬럼이 없습니다.")

    threshold_map = dict(zip(th_df["label"], th_df["best_threshold"]))

    for label in FINE_LABELS:
        if label not in threshold_map:
            raise ValueError(f"{label}의 threshold가 없습니다.")

    df = pred_df.copy()

    # label-wise threshold 적용
    for label in FINE_LABELS:
        th = float(threshold_map[label])
        prob_col = f"pred_prob__{label}"
        pred_col = f"pred_labelwise__{label}"

        df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
        df[pred_col] = (df[prob_col] >= th).astype(int)

    # 전체 metric
    y_true = df[FINE_LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
    y_pred = df[[f"pred_labelwise__{label}" for label in FINE_LABELS]].astype(int).values

    summary = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    # 라벨별 metric
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

        rows.append({
            "label": label,
            "threshold": float(threshold_map[label]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(yt.sum()),
            "pred_positive": int(yp.sum()),
            "gold_positive": int(yt.sum()),
        })

    label_df = pd.DataFrame(rows).sort_values(by=["f1", "support"], ascending=[False, False])

    # 저장
    df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    label_df.to_csv(OUT_LABEL_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nLABEL METRICS")
    print(label_df.to_string(index=False))
    print(f"\nsaved: {OUT_PRED_CSV}")
    print(f"saved: {OUT_LABEL_CSV}")
    print(f"saved: {OUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()