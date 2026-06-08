# 각 모델 단독 성능 먼저 뽑기


# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd


K_LIST = [5, 10, 20, 50, 100]

OUT_DIR = Path("outputs")
ENS_DIR = Path("ensemble_outputs")
ENS_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def find_prediction_file(model, k):
    # k=5가 k50을 잡는 문제 방지 위해 exact path 먼저 확인함
    if model == "UserCF":
        exact_candidates = [
            OUT_DIR / f"valid_user_cf_k{k}.csv",
            OUT_DIR / f"valid_user_cf_k{k} (1).csv",
            OUT_DIR / f"valid_user_cf_k{k} (2).csv",
            OUT_DIR / f"valid_user_cf_k{k} (3).csv",
        ]

        glob_patterns = [
            f"valid_user_cf_k{k} (*.csv",
            f"valid_user_cf_k{k} (*).csv",
        ]

    elif model == "MF":
        exact_candidates = [
            OUT_DIR / f"valid_mf_k{k}.csv",
            OUT_DIR / f"valid_mf_k{k} (1).csv",
            OUT_DIR / f"valid_mf_k{k} (2).csv",
            OUT_DIR / f"valid_mf_k{k} (3).csv",
        ]

        glob_patterns = [
            f"valid_mf_k{k} (*.csv",
            f"valid_mf_k{k} (*).csv",
        ]

    else:
        raise ValueError("model은 'UserCF' 또는 'MF'만 가능합니다.")

    for path in exact_candidates:
        if path.exists() and "recs" not in path.name.lower():
            return path

    matches = []
    for pattern in glob_patterns:
        matches.extend(list(OUT_DIR.glob(pattern)))

    matches = [p for p in matches if "recs" not in p.name.lower()]
    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(f"{model} k={k} 예측 파일을 찾지 못했습니다.")

    return matches[-1]


def load_pred_file(path):
    df = pd.read_csv(path, sep=None, engine="python")

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    return df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()


def evaluate_model(model_name, k, path):
    df = load_pred_file(path)

    return {
        "model": model_name,
        "k": k,
        "file": str(path),
        "rows": len(df),
        "RMSE": round(rmse(df["true_rating"], df["pred_rating"]), 4),
        "MAE": round(mae(df["true_rating"], df["pred_rating"]), 4),
    }


def main():
    rows = []

    for k in K_LIST:
        usercf_path = find_prediction_file("UserCF", k)
        mf_path = find_prediction_file("MF", k)

        print(f"\nk={k}")
        print("UserCF:", usercf_path)
        print("MF    :", mf_path)

        rows.append(evaluate_model("UserCF", k, usercf_path))
        rows.append(evaluate_model("MF", k, mf_path))

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["model", "RMSE", "MAE"]).reset_index(drop=True)

    out_path = ENS_DIR / "stage1_baseline_summary.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[Stage 1 Baseline Summary]")
    print(result_df)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
