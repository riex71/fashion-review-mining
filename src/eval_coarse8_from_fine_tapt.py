from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt" / "eval_predictions_labelwise_threshold.csv"
OUT_DIR = BASE_DIR / "outputs" / "fine_roberta_large_from_tapt"

OUT_PRED_CSV = OUT_DIR / "coarse8_predictions_from_fine.csv"
OUT_LABEL_CSV = OUT_DIR / "coarse8_label_metrics.csv"
OUT_SUMMARY_JSON = OUT_DIR / "coarse8_summary.json"

FINE_TO_COARSE = {
    "delivery": [
        "late_delivery",
        "wrong_item",
        "missing_item",
        "damaged_on_arrival",
    ],
    "packaging": [
        "packaging_issue",
    ],
    "size_fit": [
        "size_large",
        "size_small",
        "fit_dissatisfaction",
    ],
    "quality": [
        "finish_defect",
        "durability",
        "contamination",
    ],
    "fabric_comfort": [
        "fabric_issue",
        "thickness",
        "poor_comfort",
    ],
    "color_design": [
        "color_mismatch",
        "design_mismatch",
    ],
    "price": [
        "price_dissatisfaction",
    ],
    "other_complaint": [
        "other_complaint",
    ],
}

COARSE_LABELS = list(FINE_TO_COARSE.keys())


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)

    # 필요한 컬럼 체크
    for fine_labels in FINE_TO_COARSE.values():
        for label in fine_labels:
            if label not in df.columns:
                raise ValueError(f"gold fine 컬럼이 없습니다: {label}")
            pred_col = f"pred_labelwise__{label}"
            if pred_col not in df.columns:
                raise ValueError(f"pred fine 컬럼이 없습니다: {pred_col}")

    out_df = df.copy()

    # coarse gold / pred 생성
    for coarse_label, fine_labels in FINE_TO_COARSE.items():
        gold_cols = fine_labels
        pred_cols = [f"pred_labelwise__{label}" for label in fine_labels]

        out_df[coarse_label] = (
            out_df[gold_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).max(axis=1)
        )
        out_df[f"pred__{coarse_label}"] = (
            out_df[pred_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).max(axis=1)
        )

    y_true = out_df[COARSE_LABELS].astype(int).values
    y_pred = out_df[[f"pred__{label}" for label in COARSE_LABELS]].astype(int).values

    summary = {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    rows = []
    for i, label in enumerate(COARSE_LABELS):
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
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(yt.sum()),
            "pred_positive": int(yp.sum()),
            "gold_positive": int(yt.sum()),
        })

    label_df = pd.DataFrame(rows).sort_values(by=["f1", "support"], ascending=[False, False])

    out_df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
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