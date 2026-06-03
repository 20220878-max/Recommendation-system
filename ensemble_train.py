# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd


USERCF_K_LIST = [5, 10, 20, 50, 100]
MF_K_LIST = [5, 10, 20, 50, 100]
WEIGHT_LIST = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]

OUT_DIR = Path("outputs")
ENS_DIR = Path("ensemble_outputs")
ENS_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def find_prediction_file(model, k):
    if model == "UserCF":
        candidates = [
            OUT_DIR / f"valid_user_cf_k{k}.csv",
            OUT_DIR / f"valid_user_cf_k{k} (1).csv",
            OUT_DIR / f"valid_user_cf_k{k} (2).csv",
            OUT_DIR / f"valid_user_cf_k{k} (3).csv",
        ]
        glob_pattern = f"valid_user_cf_k{k} (*).csv"
    elif model == "MF":
        candidates = [
            OUT_DIR / f"valid_mf_k{k}.csv",
            OUT_DIR / f"valid_mf_k{k} (1).csv",
            OUT_DIR / f"valid_mf_k{k} (2).csv",
            OUT_DIR / f"valid_mf_k{k} (3).csv",
        ]
        glob_pattern = f"valid_mf_k{k} (*).csv"
    else:
        raise ValueError("model은 'UserCF' 또는 'MF'만 가능합니다.")

    for path in candidates:
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
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]

    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}
    missing = required_cols - set(df.columns)

    if missing and len(df.columns) == 4:
        df.columns = ["user_id", "item_id", "true_rating", "pred_rating"]

    df = df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()
    df = df.rename(columns={"pred_rating": pred_name})
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["true_rating"] = df["true_rating"].astype(float)
    df[pred_name] = df[pred_name].astype(float)

    return df


def merge_usercf_mf(usercf_k, mf_k):
    usercf_path = find_prediction_file("UserCF", usercf_k)
    mf_path = find_prediction_file("MF", mf_k)

    usercf_df = load_pred_file(usercf_path, "usercf_pred")
    mf_df = load_pred_file(mf_path, "mf_pred")

    merged = pd.merge(usercf_df, mf_df, on=["user_id", "item_id"],
                      how="inner", suffixes=("_usercf", "_mf"))

    diff = (merged["true_rating_usercf"] != merged["true_rating_mf"]).sum()
    if diff > 0:
        raise ValueError(f"true_rating 불일치: UserCF k={usercf_k}, MF k={mf_k}, diff={diff}")

    merged["true_rating"] = merged["true_rating_usercf"]
    merged = merged[["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred"]].copy()

    return merged


def evaluate(df):
    return {
        "RMSE": round(rmse(df["true_rating"], df["pred_rating"]), 4),
        "MAE": round(mae(df["true_rating"], df["pred_rating"]), 4),
        "rows": len(df)
    }


def main():
    rows = []
    best_rmse = float("inf")
    best_df = None
    best_info = None

    for usercf_k in USERCF_K_LIST:
        for mf_k in MF_K_LIST:
            merged = merge_usercf_mf(usercf_k, mf_k)
            print(f"\nPair: UserCF k={usercf_k}, MF k={mf_k} ({len(merged)} rows)")

            for w_usercf in WEIGHT_LIST:
                w_mf = round(1 - w_usercf, 1)

                temp = merged.copy()
                temp["pred_rating"] = (
                    w_usercf * temp["usercf_pred"] + w_mf * temp["mf_pred"]
                ).clip(1, 5)

                metrics = evaluate(temp)
                row = {
                    "usercf_k": usercf_k, "mf_k": mf_k,
                    "w_usercf": w_usercf, "w_mf": w_mf,
                    **metrics
                }
                rows.append(row)

                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    best_df = temp.copy()
                    best_info = row.copy()

    summary_df = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    summary_path = ENS_DIR / "ensemble_train_sweep.csv"
    best_pred_path = ENS_DIR / "ensemble_train_best_pred.csv"
    best_config_path = ENS_DIR / "ensemble_train_best_config.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best_df[["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred", "pred_rating"]].to_csv(
        best_pred_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([best_info]).to_csv(best_config_path, index=False, encoding="utf-8-sig")

    print("\n[Top 10 조합]")
    print(summary_df.head(10))

    print("\n[Best Config]")
    print(pd.DataFrame([best_info]))

    print("\n저장 완료:")
    print(f"- {summary_path}")
    print(f"- {best_pred_path}")
    print(f"- {best_config_path}")


if __name__ == "__main__":
    main()
