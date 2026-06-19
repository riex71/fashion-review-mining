from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt" / "eval_predictions.csv"
OUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt"

OUT_GLOBAL_CSV = OUT_DIR / "fine_global_threshold_metrics.csv"
OUT_LABELWISE_CSV = OUT_DIR / "fine_labelwise_best_thresholds.csv"
OUT_BEST_JSON = OUT_DIR / "fine_best_threshold_summary.json"

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


def evaluate_global(y_true, prob_df, threshold: float) -> dict:
    y_pred = (prob_df.values >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def find_best_threshold_for_label(y_true, probs, label_name: str, thresholds):
    best = None
    support = int(y_true.sum())

    for th in thresholds:
        preds = (probs >= th).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            preds,
            average="binary",
            zero_division=0,
        )

        row = {
            "label": label_name,
            "best_threshold": float(th),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "pred_positive": int(preds.sum()),
            "gold_positive": support,
        }

        if best is None:
            best = row
        else:
            if row["f1"] > best["f1"]:
                best = row
            elif row["f1"] == best["f1"] and row["recall"] > best["recall"]:
                best = row

    return best


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)

    for label in FINE_LABELS:
        if label not in df.columns:
            raise ValueError(f"gold 컬럼이 없습니다: {label}")
        if f"pred_prob__{label}" not in df.columns:
            raise ValueError(f"확률 컬럼이 없습니다: pred_prob__{label}")

    y_true = df[FINE_LABELS].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).values
    prob_df = df[[f"pred_prob__{label}" for label in FINE_LABELS]].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)

    thresholds = np.arange(0.10, 0.91, 0.05)

    global_rows = []
    for th in thresholds:
        global_rows.append(evaluate_global(y_true, prob_df, float(round(th, 2))))

    global_df = pd.DataFrame(global_rows).sort_values(
        by=["micro_f1", "macro_f1", "subset_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    global_df.to_csv(OUT_GLOBAL_CSV, index=False, encoding="utf-8-sig")

    best_global = global_df.iloc[0].to_dict()

    label_rows = []
    for label in FINE_LABELS:
        yt = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int).values
        yp = pd.to_numeric(df[f"pred_prob__{label}"], errors="coerce").fillna(0.0).values
        best = find_best_threshold_for_label(yt, yp, label, thresholds)
        label_rows.append(best)

    label_df = pd.DataFrame(label_rows).sort_values(by=["f1", "support"], ascending=[False, False])
    label_df.to_csv(OUT_LABELWISE_CSV, index=False, encoding="utf-8-sig")

    summary = {
        "best_global_threshold": best_global,
    }

    with open(OUT_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("GLOBAL THRESHOLD RESULTS")
    print(global_df.to_string(index=False))
    print("\nLABEL-WISE BEST THRESHOLDS")
    print(label_df.to_string(index=False))
    print(f"\nsaved: {OUT_GLOBAL_CSV}")
    print(f"saved: {OUT_LABELWISE_CSV}")
    print(f"saved: {OUT_BEST_JSON}")


if __name__ == "__main__":
    main()