import os
import numpy as np
import pandas as pd


DATA_DIR = 'ml-100k'
N_USERS = 943
N_ITEMS = 1682
SEED = 42
VAL_RATIO = 0.15
TOPK = 10
K_VALUES = [5, 10, 20, 50, 100]
LR = 0.005
EPOCHS = 20
REG = 0.02
REL_THRESHOLD = 4

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv(f'{DATA_DIR}/ua.base', sep='\t',
                 names=['user', 'item', 'rating', 'timestamp'])
df['user'] -= 1
df['item'] -= 1

np.random.seed(SEED)
perm = np.random.permutation(len(df))
split = int(len(df) * (1 - VAL_RATIO))
train_set = df.iloc[perm[:split]].reset_index(drop=True)
val_set   = df.iloc[perm[split:]].reset_index(drop=True)

users   = train_set['user'].values
items   = train_set['item'].values
ratings = train_set['rating'].values.astype(float)
user_rated_items = {u: g['item'].values for u, g in train_set.groupby('user')}

vu = val_set['user'].values
vi = val_set['item'].values
vr = val_set['rating'].values.astype(float)

print(f"[split] train={len(train_set)}, val={len(val_set)}  (seed={SEED}, ratio={VAL_RATIO})")

popularity_counts = np.bincount(items, minlength=N_ITEMS)
popular_topk_set  = set(np.argsort(-popularity_counts)[:TOPK].tolist())


def compute_beyond_accuracy(recommendations, Q, K):
    rec_set = set()
    for its in recommendations.values():
        rec_set.update(its)
    coverage = len(rec_set) / N_ITEMS

    ser, nov, div = [], [], []
    for its in recommendations.values():
        topk = its[:K]
        ser.append(sum(1 for i in topk if i not in popular_topk_set) / K)
        novs = [-np.log2(max(popularity_counts[i] / N_USERS, 1e-9)) for i in topk]
        nov.append(np.mean(novs))
        if len(topk) >= 2:
            vecs = Q[topk]
            normed = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            sim = normed @ normed.T
            n = len(topk)
            div.append(1 - np.sum(np.triu(sim, k=1)) / (n * (n - 1) / 2))
    return coverage, np.mean(ser), np.mean(nov), np.mean(div)


results = []
saved_k20 = {}

for k in K_VALUES:
    print(f"\nlatent_dim={k} 학습 시작")
    np.random.seed(SEED)
    P  = np.random.normal(0, 0.1, (N_USERS, k))
    Q  = np.random.normal(0, 0.1, (N_ITEMS, k))
    Y  = np.random.normal(0, 0.1, (N_ITEMS, k))
    bu = np.zeros(N_USERS)
    bi = np.zeros(N_ITEMS)
    mu = ratings.mean()

    for epoch in range(EPOCHS):
        order = np.random.permutation(len(ratings))
        for j in order:
            u, i, r = users[j], items[j], ratings[j]
            I_u = user_rated_items[u]
            norm = 1.0 / np.sqrt(len(I_u))
            p_aug = P[u] + norm * Y[I_u].sum(axis=0)
            err = r - (mu + bu[u] + bi[i] + p_aug @ Q[i])
            bu[u] += LR * (err - REG * bu[u])
            bi[i] += LR * (err - REG * bi[i])
            P[u]  += LR * (err * Q[i] - REG * P[u])
            Q[i]  += LR * (err * p_aug - REG * Q[i])
            Y[I_u] += LR * (err * norm * Q[i] - REG * Y[I_u])
    print(f"  학습 완료 ({EPOCHS} epochs)")

    P_aug = P.copy()
    for u in range(N_USERS):
        if u in user_rated_items:
            I_u = user_rated_items[u]
            P_aug[u] += (1 / np.sqrt(len(I_u))) * Y[I_u].sum(axis=0)

    val_pred = np.clip(mu + bu[vu] + bi[vi] + np.sum(P_aug[vu] * Q[vi], axis=1), 1, 5)
    pred_df = pd.DataFrame({
        'user_id': vu + 1,
        'item_id': vi + 1,
        'true_rating': vr.astype(int),
        'pred_rating': np.round(val_pred, 4),
    }).sort_values(['user_id', 'item_id']).reset_index(drop=True)
    pred_df.to_csv(f'outputs/valid_mf_k{k}.csv', index=False)

    rmse = np.sqrt(np.mean((val_pred - vr) ** 2))
    mae  = np.mean(np.abs(val_pred - vr))

    train_items_per_user = {u: set(its) for u, its in user_rated_items.items()}
    recommendations = {}
    rec_rows = []
    for u in range(N_USERS):
        scores = mu + bu[u] + bi + P_aug[u] @ Q.T
        for it in train_items_per_user.get(u, set()):
            scores[it] = -np.inf
        topk = np.argpartition(-scores, TOPK)[:TOPK]
        topk = topk[np.argsort(-scores[topk])]
        recommendations[u] = topk.tolist()
        for rank, it in enumerate(topk, 1):
            rec_rows.append([u + 1, rank, it + 1,
                             round(float(np.clip(scores[it], 1, 5)), 4)])
    recs_df = pd.DataFrame(rec_rows, columns=['user_id', 'rank', 'item_id', 'pred_rating'])
    recs_df.to_csv(f'outputs/valid_mf_recs_k{k}.csv', index=False)

    cov, ser, nov, div = compute_beyond_accuracy(recommendations, Q, TOPK)

    results.append({
        'model': 'MF', 'parameter_name': 'latent_dim', 'parameter_value': k,
        'topK': TOPK, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4),
        'Coverage': round(cov, 4), 'Serendipity': round(ser, 4),
        'Novelty': round(nov, 4), 'Diversity': round(div, 4),
    })
    print(f"  RMSE={rmse:.4f} MAE={mae:.4f} Cov={cov:.4f} "
          f"Ser={ser:.4f} Nov={nov:.4f} Div={div:.4f}")
    print(f"  저장: valid_mf_k{k}.csv, valid_mf_recs_k{k}.csv")

    if k == 20:
        saved_k20 = {'recs_df': recs_df.copy()}

results_df = pd.DataFrame(results)
results_df.to_csv('outputs/mf_results_table.csv', index=False)
print("\n[MF 결과표]")
print(results_df.to_string(index=False))
print("저장: outputs/mf_results_table.csv")


ratings_per_user = train_set.groupby('user').size()

def user_group(u):
    n = ratings_per_user.get(u, 0)
    if n < 30:    return 'Low'
    elif n < 80:  return 'Medium'
    else:         return 'High'

pred20 = pd.read_csv('outputs/valid_mf_k20.csv')
pred20['user0']   = pred20['user_id'] - 1
pred20['group']   = pred20['user0'].map(user_group)
pred20['sq_err']  = (pred20['pred_rating'] - pred20['true_rating']) ** 2
pred20['abs_err'] = (pred20['pred_rating'] - pred20['true_rating']).abs()

recs20 = saved_k20['recs_df'].copy()
recs20['user0'] = recs20['user_id'] - 1
recs20['group'] = recs20['user0'].map(user_group)

group_rows = []
for grp in ['Low', 'Medium', 'High']:
    pg = pred20[pred20['group'] == grp]
    rg = recs20[recs20['group'] == grp]
    group_rows.append({
        'model': 'MF', 'parameter_name': 'latent_dim', 'parameter_value': 20,
        'topK': TOPK, 'user_group': grp,
        'user_count': int(pg['user_id'].nunique()),
        'RMSE': round(np.sqrt(pg['sq_err'].mean()), 4),
        'MAE': round(pg['abs_err'].mean(), 4),
        'Coverage': round(rg['item_id'].nunique() / N_ITEMS, 4),
    })

group_df = pd.DataFrame(group_rows)
group_df.to_csv('outputs/mf_grouping_table.csv', index=False)
print("\n[MF Grouping (k=20)]")
print(group_df.to_string(index=False))
print("저장: outputs/mf_grouping_table.csv")

print("\n[완료] outputs/ 폴더에 모든 산출물 생성됨.")
