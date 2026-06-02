from pathlib import Path
import numpy as np
import pandas as pd


K_LIST = [5, 10, 20, 50, 100]
WEIGHT_LIST = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]  # UserCF 비중

OUT_DIR = Path("outputs")
ENS_DIR = Path("ensemble_outputs")
ENS_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def find_prediction_file(model, k):
    # k=5가 k50을 잡는 문제 방지 위해 exact path 먼저 확인
    if model == "UserCF":
        exact_candidates = [
            OUT_DIR / f"valid_user_cf_k{k}.csv",
            OUT_DIR / f"valid_user_cf_k{k} (1).csv",
            OUT_DIR / f"valid_user_cf_k{k} (2).csv",
            OUT_DIR / f"valid_user_cf_k{k} (3).csv",
        ]

        glob_pattern = f"valid_user_cf_k{k} (*).csv"

    elif model == "MF":
        exact_candidates = [
            OUT_DIR / f"valid_mf_k{k}.csv",
            OUT_DIR / f"valid_mf_k{k} (1).csv",
            OUT_DIR / f"valid_mf_k{k} (2).csv",
            OUT_DIR / f"valid_mf_k{k} (3).csv",
        ]

        glob_pattern = f"valid_mf_k{k} (*).csv"

    else:
        raise ValueError("model은 'UserCF' 또는 'MF'만 가능합니다.")

    for path in exact_candidates:
        if path.exists() and "recs" not in path.name.lower():
            return path

    matches = list(OUT_DIR.glob(glob_pattern))
    matches = [p for p in matches if "recs" not in p.name.lower()]
    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(f"{model} k={k} 예측 파일을 찾지 못했습니다.")

    return matches[-1]


def load_pred_file(path, pred_name):
    df = pd.read_csv(path, sep=None, engine="python")

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()
    df = df.rename(columns={"pred_rating": pred_name})

    return df


def merge_usercf_mf(k):
    usercf_path = find_prediction_file("UserCF", k)
    mf_path = find_prediction_file("MF", k)

    print("UserCF:", usercf_path)
    print("MF    :", mf_path)

    usercf_df = load_pred_file(usercf_path, "usercf_pred")
    mf_df = load_pred_file(mf_path, "mf_pred")

    merged = pd.merge(
        usercf_df,
        mf_df,
        on=["user_id", "item_id"],
        how="inner",
        suffixes=("_usercf", "_mf")
    )

    diff = (merged["true_rating_usercf"] != merged["true_rating_mf"]).sum()
    if diff > 0:
        raise ValueError(f"k={k}: true_rating 불일치 {diff}개 발견")

    merged["true_rating"] = merged["true_rating_usercf"]
    merged = merged[["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred"]].copy()

    return merged


def make_weighted_prediction(df, w_user):
    out = df.copy()
    out["pred_rating"] = w_user * out["usercf_pred"] + (1 - w_user) * out["mf_pred"]
    out["pred_rating"] = out["pred_rating"].clip(1, 5)
    return out


def evaluate(df):
    return {
        "RMSE": round(rmse(df["true_rating"], df["pred_rating"]), 4),
        "MAE": round(mae(df["true_rating"], df["pred_rating"]), 4),
        "rows": len(df)
    }


def main():
    summary_rows = []

    best_rmse = float("inf")
    best_df = None
    best_info = None

    for k in K_LIST:
        print(f"\nStage 3 Weighted Ensemble k={k}")
        merged = merge_usercf_mf(k)
        print("merged rows:", len(merged))

        for w_user in WEIGHT_LIST:
            w_mf = round(1 - w_user, 1)

            weighted_df = make_weighted_prediction(merged, w_user)
            metrics = evaluate(weighted_df)

            row = {
                "model": "Weighted_Ensemble",
                "k": k,
                "w_usercf": w_user,
                "w_mf": w_mf,
                **metrics
            }

            summary_rows.append(row)

            print(
                f"w_usercf={w_user:.1f}, w_mf={w_mf:.1f} | "
                f"RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}"
            )

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_df = weighted_df.copy()
                best_info = row.copy()

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    summary_path = ENS_DIR / "stage3_weighted_ensemble_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_pred_path = ENS_DIR / "valid_stage3_weighted_ensemble_best.csv"
    best_df[
        ["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred", "pred_rating"]
    ].to_csv(best_pred_path, index=False, encoding="utf-8-sig")

    best_config_df = pd.DataFrame([best_info])
    best_config_path = ENS_DIR / "stage3_weighted_ensemble_best_config.csv"
    best_config_df.to_csv(best_config_path, index=False, encoding="utf-8-sig")

    print("\n[Weighted Ensemble Summary Top 10]")
    print(summary_df.head(10))

    print("\n[Best Config]")
    print(best_config_df)

    print("\n저장 완료:")
    print(f"- {summary_path}")
    print(f"- {best_pred_path}")
    print(f"- {best_config_path}")


if __name__ == "__main__":
    main()
