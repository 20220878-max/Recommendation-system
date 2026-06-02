from pathlib import Path
import numpy as np
import pandas as pd


TOP_K = 10

OUT_DIR = Path("outputs")
ENS_DIR = Path("ensemble_outputs")
RAW_DIR = Path("raw/ml-100k")

UA_BASE_PATH = RAW_DIR / "ua.base"
U_ITEM_PATH = RAW_DIR / "u.item"

TOTAL_ITEMS = 1682


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def evaluate_accuracy(df):
    return {
        "RMSE": round(rmse(df["true_rating"], df["pred_rating"]), 4),
        "MAE": round(mae(df["true_rating"], df["pred_rating"]), 4),
    }


def find_usercf_k50():
    candidates = [
        OUT_DIR / "valid_user_cf_k50.csv",
        OUT_DIR / "valid_user_cf_k50 (1).csv",
        OUT_DIR / "valid_user_cf_k50 (2).csv",
        OUT_DIR / "valid_user_cf_k50 (3).csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("UserCF k=50 파일을 찾지 못했습니다.")


def find_mf_k50():
    candidates = [
        OUT_DIR / "valid_mf_k50.csv",
        OUT_DIR / "valid_mf_k50 (1).csv",
        OUT_DIR / "valid_mf_k50 (2).csv",
        OUT_DIR / "valid_mf_k50 (3).csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("MF k=50 파일을 찾지 못했습니다.")


def normalize_column_name(col):
    return str(col).replace("﻿", "").strip().lower()


def load_prediction_file(path):
    df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    rename_map = {}
    for col in df.columns:
        lower = normalize_column_name(col)
        if lower in ["user_id", "userid", "user", "user id", "uid"]:
            rename_map[col] = "user_id"
        elif lower in ["item_id", "itemid", "item", "item id", "iid", "movie_id", "movieid", "movie"]:
            rename_map[col] = "item_id"
        elif lower in ["true_rating", "true rating", "true", "rating", "actual_rating", "actual"]:
            rename_map[col] = "true_rating"
        elif lower in ["pred_rating", "pred rating", "pred", "prediction", "estimated_rating", "estimate", "est"]:
            rename_map[col] = "pred_rating"

    df = df.rename(columns=rename_map)

    required_cols = {"user_id", "item_id", "true_rating", "pred_rating"}
    missing = required_cols - set(df.columns)

    if missing and len(df.columns) == 4:
        print(f"[주의] {path} 컬럼명을 순서 기준으로 보정합니다.")
        print("기존 컬럼:", df.columns.tolist())
        df.columns = ["user_id", "item_id", "true_rating", "pred_rating"]
        missing = required_cols - set(df.columns)

    if missing:
        print(f"\n[컬럼 확인 필요] {path}")
        print("현재 컬럼:", df.columns.tolist())
        print(df.head())
        raise ValueError(f"{path}에 필요한 컬럼이 없습니다: {missing}")

    df = df[["user_id", "item_id", "true_rating", "pred_rating"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["true_rating"] = df["true_rating"].astype(float)
    df["pred_rating"] = df["pred_rating"].astype(float)

    return df


def load_ua_base():
    if not UA_BASE_PATH.exists():
        raise FileNotFoundError("raw/ml-100k/ua.base 파일이 없습니다.")

    df = pd.read_csv(
        UA_BASE_PATH,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


def load_item_genres():
    if not U_ITEM_PATH.exists():
        raise FileNotFoundError("raw/ml-100k/u.item 파일이 없습니다.")

    item_df = pd.read_csv(U_ITEM_PATH, sep="|", header=None, encoding="latin-1")

    genre_cols = list(range(5, item_df.shape[1]))
    genre_dict = {}

    for _, row in item_df.iterrows():
        item_id = int(row[0])
        genre_dict[item_id] = row[genre_cols].to_numpy(dtype=float)

    return genre_dict


def make_recommendations(df, top_k=10):
    rec_df = (
        df.sort_values(["user_id", "pred_rating"], ascending=[True, False])
        .groupby("user_id")
        .head(top_k)
    )

    return (
        rec_df.groupby("user_id")["item_id"]
        .apply(lambda x: list(x.astype(int)))
        .to_dict()
    )


def coverage(recs, n_items=1682):
    rec_set = set()
    for items in recs.values():
        rec_set.update(items)
    return len(rec_set) / n_items


def novelty(recs, popularity_counts, n_users, top_k=10):
    # 추천 아이템의 평균 self-information (-log2 popularity probability)
    user_scores = []

    for items in recs.values():
        item_scores = []
        for item_id in items[:top_k]:
            pop = popularity_counts.get(item_id, 0)
            prob = max(pop / n_users, 1e-9)
            item_scores.append(-np.log2(prob))
        if item_scores:
            user_scores.append(np.mean(item_scores))

    return np.mean(user_scores)


def serendipity(recs, popular_set, top_k=10):
    # 추천 목록 중 인기 baseline에 없는 아이템 비율
    scores = []

    for items in recs.values():
        top_items = items[:top_k]
        if not top_items:
            continue
        scores.append(sum(1 for item_id in top_items if item_id not in popular_set) / len(top_items))

    return np.mean(scores)


def diversity(recs, genre_dict, top_k=10):
    # 추천 리스트 내 아이템 간 평균 비유사도 (장르 cosine 기반)
    scores = []

    for items in recs.values():
        vecs = [genre_dict[i] for i in items[:top_k] if i in genre_dict]

        if len(vecs) < 2:
            continue

        vecs = np.array(vecs, dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        normed = vecs / norms
        sim_matrix = normed @ normed.T

        n = len(vecs)
        pair_count = n * (n - 1) / 2

        if pair_count == 0:
            continue

        avg_sim = np.sum(np.triu(sim_matrix, k=1)) / pair_count
        scores.append(1 - avg_sim)

    if not scores:
        return 0.0

    return np.mean(scores)


def evaluate_model(model_name, path, ua_base_df, genre_dict):
    print(f"\nEvaluating {model_name}")
    print(f"file: {path}")

    df = load_prediction_file(path)
    acc = evaluate_accuracy(df)
    recs = make_recommendations(df, top_k=TOP_K)

    popularity_counts = ua_base_df.groupby("item_id").size().to_dict()
    n_users = ua_base_df["user_id"].nunique()

    popular_set = set(
        ua_base_df.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .head(TOP_K)
        .index
        .astype(int)
        .tolist()
    )

    non_acc = {
        "Coverage": round(coverage(recs, TOTAL_ITEMS), 4),
        "Novelty": round(novelty(recs, popularity_counts, n_users, TOP_K), 4),
        "Serendipity": round(serendipity(recs, popular_set, TOP_K), 4),
        "Diversity": round(diversity(recs, genre_dict, TOP_K), 4),
    }

    return {
        "model": model_name,
        "file": str(path),
        "topK": TOP_K,
        "users": len(recs),
        **acc,
        **non_acc
    }


def main():
    ua_base_df = load_ua_base()
    genre_dict = load_item_genres()

    model_files = {
        "UserCF_best_k50": find_usercf_k50(),
        "MF_best_k50": find_mf_k50(),
        "TrueBagging_UserCF_k50": ENS_DIR / "valid_true_bagging_usercf_k50.csv",
        "Weighted_Ensemble_best": ENS_DIR / "valid_stage3_weighted_ensemble_best.csv",
        "BoostingStyle_UserCF": ENS_DIR / "valid_stage4_boosting_usercf.csv",
    }

    rows = []

    for model_name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{model_name} 파일이 없습니다: {path}")
        rows.append(evaluate_model(model_name, path, ua_base_df, genre_dict))

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    out_path = ENS_DIR / "stage5_final_metrics_summary.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n[Stage 5 Final Metrics Summary]")
    print(result_df)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
