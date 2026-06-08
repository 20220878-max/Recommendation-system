# 배깅 - 부트스트랩으로 25개 모델 돌리기


# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, KNNBasic


SEED = 42
TRAIN_RATIO = 0.85
K_LIST = [5, 10, 20, 50, 100]
N_BAGS = 5  # 시간이 너무 오래 걸리면 3으로 줄여도 됨

ENS_DIR = Path("ensemble_outputs")
ENS_DIR.mkdir(exist_ok=True)


def find_ua_base():
    candidates = [
        Path("raw/ml-100k/ua.base"),
        Path("ml-100k/ua.base"),
        Path("data/ml-100k/ua.base"),
        Path("ua.base"),
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "ua.base 파일을 찾지 못했습니다. "
        "raw/ml-100k/ua.base 또는 ml-100k/ua.base 위치를 확인하세요."
    )


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def load_and_split():
    ua_base_path = find_ua_base()
    print(f"ua.base path: {ua_base_path}")

    df = pd.read_csv(
        ua_base_path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    np.random.seed(SEED)
    idx = np.random.permutation(len(df))

    train_size = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[idx[:train_size]].reset_index(drop=True)
    val_df = df.iloc[idx[train_size:]].reset_index(drop=True)

    print(f"total rows      : {len(df)}")
    print(f"train rows      : {len(train_df)}")
    print(f"validation rows : {len(val_df)}")

    return train_df, val_df


def train_usercf(train_df, k):
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    algo = KNNBasic(
        k=k,
        sim_options={"name": "cosine", "user_based": True},
        verbose=False
    )

    algo.fit(trainset)
    return algo


def predict_validation(algo, val_df):
    preds = []

    for row in val_df.itertuples(index=False):
        pred = algo.predict(uid=row.user_id, iid=row.item_id, r_ui=row.rating)
        preds.append(pred.est)

    return np.array(preds)


def run_true_bagging_usercf(train_df, val_df, k):
    bag_predictions = []

    for bag_id in range(N_BAGS):
        print(f"  bootstrap model {bag_id + 1}/{N_BAGS}")

        boot_df = train_df.sample(
            n=len(train_df),
            replace=True,
            random_state=SEED + bag_id
        ).reset_index(drop=True)

        algo = train_usercf(boot_df, k)
        pred = predict_validation(algo, val_df)
        bag_predictions.append(pred)

    final_pred = np.clip(np.vstack(bag_predictions).mean(axis=0), 1, 5)

    pred_df = pd.DataFrame({
        "user_id": val_df["user_id"],
        "item_id": val_df["item_id"],
        "true_rating": val_df["rating"],
        "pred_rating": final_pred
    })

    metrics = {
        "model": "TrueBagging_UserCF",
        "base_model": "UserCF_KNNBasic_cosine",
        "k": k,
        "n_bags": N_BAGS,
        "rows": len(pred_df),
        "RMSE": round(rmse(pred_df["true_rating"], pred_df["pred_rating"]), 4),
        "MAE": round(mae(pred_df["true_rating"], pred_df["pred_rating"]), 4)
    }

    return pred_df, metrics


def main():
    train_df, val_df = load_and_split()

    summary_rows = []

    for k in K_LIST:
        print(f"\nTrue Bagging UserCF k={k}")

        pred_df, metrics = run_true_bagging_usercf(train_df, val_df, k)

        pred_path = ENS_DIR / f"valid_true_bagging_usercf_k{k}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        summary_rows.append(metrics)
        print(metrics)
        print(f"saved: {pred_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    summary_path = ENS_DIR / "stage2_true_bagging_usercf_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[Stage 2 True Bagging Summary]")
    print(summary_df)
    print(f"\n저장 완료: {summary_path}")


if __name__ == "__main__":
    main()
