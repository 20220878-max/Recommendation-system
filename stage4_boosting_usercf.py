# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, KNNBasic


SEED = 42
TRAIN_RATIO = 0.85
K = 50  # Stage 1~3에서 k=50이 가장 좋았으므로
N_ROUNDS = 5
BOOST_FACTOR = 1.5

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

    raise FileNotFoundError("ua.base 파일을 찾지 못했습니다.")


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


def predict_df(algo, df):
    preds = []

    for row in df.itertuples(index=False):
        pred = algo.predict(uid=row.user_id, iid=row.item_id, r_ui=row.rating)
        preds.append(pred.est)

    return np.array(preds)


def run_boosting_usercf(train_df, val_df):
    rng = np.random.default_rng(SEED)
    sample_weights = np.ones(len(train_df)) / len(train_df)

    val_predictions = []
    round_rows = []

    for round_id in range(1, N_ROUNDS + 1):
        print(f"\nBoosting round {round_id}/{N_ROUNDS}")

        sampled_idx = rng.choice(len(train_df), size=len(train_df), replace=True, p=sample_weights)
        sampled_train = train_df.iloc[sampled_idx].reset_index(drop=True)

        algo = train_usercf(sampled_train, K)

        train_pred = predict_df(algo, train_df)
        train_true = train_df["rating"].to_numpy()
        abs_error = np.abs(train_true - train_pred)

        train_rmse = rmse(train_true, train_pred)
        train_mae = mae(train_true, train_pred)

        val_pred = predict_df(algo, val_df)
        val_predictions.append(val_pred)

        val_true = val_df["rating"].to_numpy()
        val_rmse = rmse(val_true, val_pred)
        val_mae = mae(val_true, val_pred)

        print(f"Train RMSE={train_rmse:.4f}, Train MAE={train_mae:.4f}")
        print(f"Val   RMSE={val_rmse:.4f}, Val   MAE={val_mae:.4f}")

        threshold = np.median(abs_error)
        hard_mask = abs_error > threshold
        sample_weights[hard_mask] *= BOOST_FACTOR
        sample_weights = sample_weights / sample_weights.sum()

        round_rows.append({
            "round": round_id,
            "k": K,
            "boost_factor": BOOST_FACTOR,
            "error_threshold_median": round(threshold, 4),
            "hard_samples": int(hard_mask.sum()),
            "train_RMSE": round(train_rmse, 4),
            "train_MAE": round(train_mae, 4),
            "val_RMSE": round(val_rmse, 4),
            "val_MAE": round(val_mae, 4)
        })

    val_predictions = np.vstack(val_predictions)
    final_pred = np.clip(val_predictions.mean(axis=0), 1, 5)

    result_df = pd.DataFrame({
        "user_id": val_df["user_id"],
        "item_id": val_df["item_id"],
        "true_rating": val_df["rating"],
        "pred_rating": final_pred
    })

    final_metrics = {
        "model": "BoostingStyle_UserCF",
        "base_model": "UserCF_KNNBasic_cosine",
        "k": K,
        "n_rounds": N_ROUNDS,
        "boost_factor": BOOST_FACTOR,
        "rows": len(result_df),
        "RMSE": round(rmse(result_df["true_rating"], result_df["pred_rating"]), 4),
        "MAE": round(mae(result_df["true_rating"], result_df["pred_rating"]), 4)
    }

    return result_df, pd.DataFrame(round_rows), final_metrics


def main():
    train_df, val_df = load_and_split()

    pred_df, round_df, final_metrics = run_boosting_usercf(train_df, val_df)

    pred_path = ENS_DIR / "valid_stage4_boosting_usercf.csv"
    round_path = ENS_DIR / "stage4_boosting_usercf_rounds.csv"
    summary_path = ENS_DIR / "stage4_boosting_usercf_summary.csv"

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    round_df.to_csv(round_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([final_metrics]).to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[Stage 4 Boosting Round Summary]")
    print(round_df)

    print("\n[Stage 4 Final Summary]")
    print(pd.DataFrame([final_metrics]))

    print("\n저장 완료:")
    print(f"- {pred_path}")
    print(f"- {round_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
