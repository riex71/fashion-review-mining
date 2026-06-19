from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "outputs" / "external_predictions"
OUTPUT_DIR = BASE_DIR / "outputs" / "external_manual_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLATFORMS = ["ably", "zigzag"]
TOP_K = 5

MIN_LEN = 20
MAX_LEN = 80


def add_label_text(df: pd.DataFrame) -> pd.DataFrame:
    fine_cols = [c for c in df.columns if c.startswith("pred_labelwise__")]

    def fine_list(row):
        return ", ".join(
            c.replace("pred_labelwise__", "")
            for c in fine_cols
            if int(row[c]) == 1
        )

    df = df.copy()
    df["content_len"] = df["content_norm"].fillna("").astype(str).str.len()
    df["predicted_fine_labels"] = df.apply(fine_list, axis=1)
    df["manual_check"] = ""
    df["manual_notes"] = ""
    return df


def pick_top5(df: pd.DataFrame) -> pd.DataFrame:
    df = add_label_text(df)

    # 짧고, complaint이고, 확신 높은 것
    df = df[
        (df["pred_complaint"] == 1)
        & (df["content_len"] >= MIN_LEN)
        & (df["content_len"] <= MAX_LEN)
    ].copy()

    # 라벨 2개 이상 우선, 그다음 확신 높은 순
    df = df.sort_values(
        by=["pred_fine_label_count", "max_fine_prob", "content_len"],
        ascending=[False, False, True],
    )

    # 같은 상품 중복 제거
    if "goods_no" in df.columns:
        df = df.drop_duplicates(subset=["goods_no"], keep="first")

    return df.head(TOP_K).copy()


def main():
    for platform in PLATFORMS:
        path = INPUT_DIR / f"{platform}_predictions.csv"
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

        sample_df = pick_top5(df)

        keep_cols = [
            "platform",
            "goods_name",
            "analysis_category",
            "content_norm",
            "content_len",
            "predicted_fine_labels",
            "pred_fine_label_count",
            "max_fine_prob",
            "manual_check",
            "manual_notes",
        ]
        keep_cols = [c for c in keep_cols if c in sample_df.columns]

        out_path = OUTPUT_DIR / f"{platform}_manual_review_top5.csv"
        sample_df[keep_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n[{platform}] saved: {out_path}")
        print(sample_df[keep_cols].to_string(index=False))


if __name__ == "__main__":
    main()