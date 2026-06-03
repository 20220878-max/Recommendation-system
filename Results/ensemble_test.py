# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd


USERCF_TEST_PATH = Path("Results/test_user_cf_k50 (1).csv")
MF_TEST_PATH     = Path("Results/test_mf_k50_final.csv")
CONFIG_PATH      = Path("ensemble_outputs/ensemble_train_best_config.csv")

OUT_DIR = Path("Results")
OUT_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def load_pred_file(path):
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
    df = df.astype({"user_id": int, "item_id": int,
                    "true_rating": float, "pred_rating": float})
    return df


def main():
    # 최적 가중치 로드 (ensemble_train.py에서 저장한 것)
    config = pd.read_csv(CONFIG_PATH).iloc[0]
    w_usercf = config["w_usercf"]
    w_mf     = config["w_mf"]

    print(f"Best config: UserCF k=50 w={w_usercf}, MF k=50 w={w_mf}")

    # test prediction 파일 로드
    usercf_df = load_pred_file(USERCF_TEST_PATH).rename(columns={"pred_rating": "usercf_pred"})
    mf_df     = load_pred_file(MF_TEST_PATH).rename(columns={"pred_rating": "mf_pred"})

    merged = pd.merge(usercf_df, mf_df, on=["user_id", "item_id"],
                      how="inner", suffixes=("_usercf", "_mf"))

    diff = (merged["true_rating_usercf"] != merged["true_rating_mf"]).sum()
    if diff > 0:
        raise ValueError(f"true_rating 불일치 {diff}개 발견")

    merged["true_rating"] = merged["true_rating_usercf"]
    merged = merged[["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred"]].copy()

    # 앙상블 예측
    merged["pred_rating"] = (
        w_usercf * merged["usercf_pred"] + w_mf * merged["mf_pred"]
    ).clip(1, 5)

    # 최종 성능
    r = round(rmse(merged["true_rating"], merged["pred_rating"]), 4)
    m = round(mae(merged["true_rating"],  merged["pred_rating"]), 4)

    print(f"\n[Weighted Ensemble Test 결과]")
    print(f"rows  : {len(merged)}")
    print(f"RMSE  : {r}")
    print(f"MAE   : {m}")

    # 저장
    out_path = OUT_DIR / "test_ensemble_pred.csv"
    merged[["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred", "pred_rating"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
