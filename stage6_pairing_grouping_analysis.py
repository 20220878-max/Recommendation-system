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
RAW_DIR = Path("raw/ml-100k")
UA_BASE_PATH = RAW_DIR / "ua.base"

ENS_DIR.mkdir(exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def evaluate(df, pred_col="pred_rating"):
    return {
        "RMSE": round(rmse(df["true_rating"], df[pred_col]), 4),
        "MAE": round(mae(df["true_rating"], df[pred_col]), 4),
        "rows": len(df)
    }


def find_prediction_file(model, k):
    # k=5가 k50을 잡는 문제 방지 위해 exact path 먼저 확인

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
        raise ValueError("model은 UserCF 또는 MF만 가능합니다.")

    for path in candidates:
        if path.exists() and "recs" not in path.name.lower():
            return path

    matches = list(OUT_DIR.glob(glob_pattern))

    matches = [
        p for p in matches
        if "recs" not in p.name.lower()
    ]

    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(f"{model} k={k} 파일을 찾지 못했습니다.")

    return matches[-1]


def load_pred_file(path, pred_name):
    df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()
    df = df.rename(columns={"pred_rating": pred_name})

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["true_rating"] = df["true_rating"].astype(float)
    df[pred_name] = df[pred_name].astype(float)

    return df


def load_standard_pred_file(path):
    df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}

    if not required_cols.issubset(set(df.columns)) and len(df.columns) == 4:
        df.columns = ["user_id", "item_id", "true_rating", "pred_rating"]

    missing = required_cols - set(df.columns)

    if missing:
        print(path)
        print(df.columns.tolist())
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["true_rating"] = df["true_rating"].astype(float)
    df["pred_rating"] = df["pred_rating"].astype(float)

    return df


def merge_usercf_mf(usercf_k, mf_k):
    usercf_path = find_prediction_file("UserCF", usercf_k)
    mf_path = find_prediction_file("MF", mf_k)

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
        raise ValueError(
            f"true_rating 불일치 발견: UserCF k={usercf_k}, MF k={mf_k}, diff={diff}"
        )

    merged["true_rating"] = merged["true_rating_usercf"]

    merged = merged[
        ["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred"]
    ].copy()

    return merged


def run_all_pair_weight_sweep():
    rows = []

    best_rmse = float("inf")
    best_df = None
    best_info = None

    mf20_usercf50_rows = []

    for usercf_k in USERCF_K_LIST:
        for mf_k in MF_K_LIST:
            merged = merge_usercf_mf(usercf_k, mf_k)
            print(f"\nPair: UserCF k={usercf_k}, MF k={mf_k} ({len(merged)} rows)")

            for w_usercf in WEIGHT_LIST:
                w_mf = round(1 - w_usercf, 1)

                temp = merged.copy()

                temp["pred_rating"] = (
                    w_usercf * temp["usercf_pred"]
                    + w_mf * temp["mf_pred"]
                )

                temp["pred_rating"] = temp["pred_rating"].clip(1, 5)

                metrics = evaluate(temp)

                row = {
                    "model": "AllPair_Weighted_Ensemble",
                    "usercf_k": usercf_k,
                    "mf_k": mf_k,
                    "w_usercf": w_usercf,
                    "w_mf": w_mf,
                    **metrics
                }

                rows.append(row)

                if usercf_k == 50 and mf_k == 20:
                    mf20_usercf50_rows.append(row.copy())

                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    best_df = temp.copy()
                    best_info = row.copy()

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    summary_path = ENS_DIR / "stage6_all_pair_weight_sweep.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    best_pred_path = ENS_DIR / "valid_stage6_all_pair_weighted_best.csv"
    best_df[
        ["user_id", "item_id", "true_rating", "usercf_pred", "mf_pred", "pred_rating"]
    ].to_csv(best_pred_path, index=False, encoding="utf-8-sig")

    best_config_path = ENS_DIR / "stage6_all_pair_best_config.csv"
    pd.DataFrame([best_info]).to_csv(best_config_path, index=False, encoding="utf-8-sig")

    mf20_usercf50_df = pd.DataFrame(mf20_usercf50_rows)
    mf20_usercf50_df = mf20_usercf50_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    mf20_usercf50_path = ENS_DIR / "stage6_mf20_usercf50_weight_sweep.csv"
    mf20_usercf50_df.to_csv(mf20_usercf50_path, index=False, encoding="utf-8-sig")

    print("\n[All Pair Weight Sweep Top 15]")
    print(summary_df.head(15))

    print("\n[Best Config]")
    print(pd.DataFrame([best_info]))

    print("\n[MF k=20 + UserCF k=50 Sweep]")
    print(mf20_usercf50_df)

    print("\n저장 완료:")
    print(f"- {summary_path}")
    print(f"- {best_pred_path}")
    print(f"- {best_config_path}")
    print(f"- {mf20_usercf50_path}")

    return best_pred_path


def load_user_groups():
    # rating 수 기준 3분위로 cold/medium/active 분류
    if not UA_BASE_PATH.exists():
        raise FileNotFoundError("raw/ml-100k/ua.base 파일이 없습니다.")

    ua_base = pd.read_csv(
        UA_BASE_PATH,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    user_counts = (
        ua_base.groupby("user_id")
        .size()
        .reset_index(name="rating_count")
    )

    q1 = user_counts["rating_count"].quantile(1 / 3)
    q2 = user_counts["rating_count"].quantile(2 / 3)

    def assign_group(count):
        if count <= q1:
            return "cold"
        elif count <= q2:
            return "medium"
        else:
            return "active"

    user_counts["group"] = user_counts["rating_count"].apply(assign_group)

    print(f"\nUser group: q1={q1:.2f}, q2={q2:.2f}")
    print(user_counts["group"].value_counts())

    return user_counts[["user_id", "rating_count", "group"]]


def evaluate_by_group(model_name, pred_path, user_groups):
    df = load_standard_pred_file(pred_path)

    df = df.merge(
        user_groups,
        on="user_id",
        how="left"
    )

    rows = []

    for group_name, group_df in df.groupby("group"):
        if len(group_df) == 0:
            continue

        metrics = evaluate(group_df)

        rows.append({
            "model": model_name,
            "group": group_name,
            "users": group_df["user_id"].nunique(),
            "avg_rating_count": round(group_df["rating_count"].mean(), 2),
            **metrics
        })

    # 전체도 추가
    metrics_all = evaluate(df)

    rows.append({
        "model": model_name,
        "group": "all",
        "users": df["user_id"].nunique(),
        "avg_rating_count": round(df["rating_count"].mean(), 2),
        **metrics_all
    })

    return rows


def run_grouping_analysis(all_pair_best_path):
    user_groups = load_user_groups()

    model_files = {
        "UserCF_best_k50": find_prediction_file("UserCF", 50),
        "MF_best_k50": find_prediction_file("MF", 50),
        "TrueBagging_UserCF_k50": ENS_DIR / "valid_true_bagging_usercf_k50.csv",
        "Weighted_same_k_best": ENS_DIR / "valid_stage3_weighted_ensemble_best.csv",
        "Weighted_all_pair_best": all_pair_best_path,
        "BoostingStyle_UserCF": ENS_DIR / "valid_stage4_boosting_usercf.csv",
    }

    rows = []

    for model_name, path in model_files.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{model_name} 파일이 없습니다: {path}")

        print(f"\nGrouping evaluating: {model_name}")
        rows.extend(evaluate_by_group(model_name, path, user_groups))

    result_df = pd.DataFrame(rows)

    group_order = {"cold": 0, "medium": 1, "active": 2, "all": 3}
    result_df["group_order"] = result_df["group"].map(group_order)

    result_df = result_df.sort_values(
        ["group_order", "RMSE", "MAE"]
    ).drop(columns=["group_order"]).reset_index(drop=True)

    out_path = ENS_DIR / "stage6_grouping_summary.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[Grouping Summary]")
    print(result_df.to_string(index=False))

    print(f"\n저장 완료: {out_path}")


def main():
    all_pair_best_path = run_all_pair_weight_sweep()
    run_grouping_analysis(all_pair_best_path)


if __name__ == "__main__":
    main()