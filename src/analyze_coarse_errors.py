from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = BASE_DIR / "outputs" / "coarse_roberta_large" / "eval_predictions.csv"
OUT_DIR = BASE_DIR / "outputs" / "coarse_roberta_large"

FP_CSV = OUT_DIR / "errors_false_positive.csv"
FN_CSV = OUT_DIR / "errors_false_negative.csv"
HARD_CSV = OUT_DIR / "errors_hard_cases.csv"


def main():
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {PRED_CSV}")

    df = pd.read_csv(PRED_CSV, encoding="utf-8-sig", low_memory=False)

    required_cols = ["sample_id", "content_norm", "complaint", "pred_prob", "pred_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    df = df.copy()
    df["complaint"] = pd.to_numeric(df["complaint"], errors="coerce").fillna(0).astype(int)
    df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="coerce")

    fp_df = df[(df["complaint"] == 0) & (df["pred_label"] == 1)].copy()
    fn_df = df[(df["complaint"] == 1) & (df["pred_label"] == 0)].copy()

    # 애매한 샘플: 확률이 0.4~0.6 부근
    hard_df = df[(df["pred_prob"] >= 0.4) & (df["pred_prob"] <= 0.6)].copy()

    show_cols = [
        "sample_id",
        "complaint",
        "pred_label",
        "pred_prob",
        "content_norm",
        "goods_name",
        "analysis_category",
        "option_text",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    fp_df = fp_df[show_cols].sort_values("pred_prob", ascending=False).reset_index(drop=True)
    fn_df = fn_df[show_cols].sort_values("pred_prob", ascending=True).reset_index(drop=True)
    hard_df = hard_df[show_cols].sort_values("pred_prob", ascending=True).reset_index(drop=True)

    fp_df.to_csv(FP_CSV, index=False, encoding="utf-8-sig")
    fn_df.to_csv(FN_CSV, index=False, encoding="utf-8-sig")
    hard_df.to_csv(HARD_CSV, index=False, encoding="utf-8-sig")

    print("DONE")
    print(f"FP rows: {len(fp_df)}")
    print(f"FN rows: {len(fn_df)}")
    print(f"HARD rows: {len(hard_df)}")
    print(f"saved: {FP_CSV}")
    print(f"saved: {FN_CSV}")
    print(f"saved: {HARD_CSV}")


if __name__ == "__main__":
    main()