from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = BASE_DIR / "outputs" / "coarse_roberta_large" / "eval_predictions.csv"
OUT_DIR = BASE_DIR / "outputs" / "coarse_roberta_large"
OUT_CSV = OUT_DIR / "threshold_metrics.csv"
OUT_JSON = OUT_DIR / "best_threshold_metrics.json"


def evaluate_at_threshold(y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pred_positive_rate": float(preds.mean()),
    }


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)

    required_cols = ["complaint", "pred_prob"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    y_true = pd.to_numeric(df["complaint"], errors="coerce").fillna(0).astype(int).values
    probs = pd.to_numeric(df["pred_prob"], errors="coerce").values

    thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []
    for th in thresholds:
        rows.append(evaluate_at_threshold(y_true, probs, float(round(th, 2))))

    result_df = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best_row = result_df.iloc[0].to_dict()

    result_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(result_df.to_string(index=False))
    print("\nBEST")
    print(json.dumps(best_row, ensure_ascii=False, indent=2))
    print(f"\nsaved: {OUT_CSV}")
    print(f"saved: {OUT_JSON}")


if __name__ == "__main__":
    main()