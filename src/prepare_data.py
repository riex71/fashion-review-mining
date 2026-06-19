from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "silverset.csv"

OUT_TRAIN_CSV = DATA_DIR / "train_coarse.csv"
OUT_EVAL_CSV = DATA_DIR / "eval_coarse.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"파일이 없습니다: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", low_memory=False)

    required_cols = ["sample_id", "data_source", "content_norm", "complaint"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 컬럼이 없습니다: {col}")

    df = df.copy()
    df["content_norm"] = df["content_norm"].fillna("").astype(str).str.strip()
    df["complaint"] = pd.to_numeric(df["complaint"], errors="coerce").fillna(0).astype(int)

    df = df[df["content_norm"] != ""].copy()

    train_df = df[df["data_source"] == "silver30k"].copy()
    eval_df = df[df["data_source"] == "gold500"].copy()

    keep_cols = [
        "sample_id",
        "data_source",
        "content_norm",
        "complaint",
        "goods_name",
        "analysis_category",
        "option_text",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    train_df = train_df[keep_cols].reset_index(drop=True)
    eval_df = eval_df[keep_cols].reset_index(drop=True)

    train_df.to_csv(OUT_TRAIN_CSV, index=False, encoding="utf-8-sig")
    eval_df.to_csv(OUT_EVAL_CSV, index=False, encoding="utf-8-sig")

    print("DONE")
    print(f"train rows: {len(train_df)}")
    print(f"eval rows: {len(eval_df)}")
    print(f"train complaint rate: {train_df['complaint'].mean():.4f}")
    print(f"eval complaint rate: {eval_df['complaint'].mean():.4f}")
    print(f"saved: {OUT_TRAIN_CSV}")
    print(f"saved: {OUT_EVAL_CSV}")


if __name__ == "__main__":
    main()