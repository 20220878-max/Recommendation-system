"""
mf_generate_outputs.py
======================
모델기반 CF (SVD++) — 앙상블 팀 요청 산출물 일괄 생성

생성물 (모두 outputs/ 폴더에):
  1. mf_results_table.csv      : k별 결과표 (RMSE/MAE/Coverage/Serendipity/Novelty/Diversity)
  2. valid_mf_k{5,10,20,50,100}.csv      : validation 예측 (user_id,item_id,true_rating,pred_rating)
  3. valid_mf_recs_k{5,10,20,50,100}.csv : top-10 추천 리스트 (user_id,rank,item_id,pred_rating)
  4. mf_grouping_table.csv     : k=20 사용자 활동량 그룹별 (Low/Medium/High) 성능

실행:  python mf_generate_outputs.py
       (ml-100k 폴더가 같은 위치에 있어야 함)

⚠️ 팀 통일 기준 (앙상블 머지를 위해 모든 팀원이 동일하게):
   - train:val = 85:15
   - SEED = 42
   - split 방식 = 전체 ua.base를 np.random.permutation으로 셔플 후 앞 85% / 뒤 15%
   - 이 split 코드를 팀원들과 공유할 것!
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# 0. 설정 (★ 팀 통일 기준 ★)
# ============================================================
DATA_DIR   = 'ml-100k'
N_USERS    = 943
N_ITEMS    = 1682
SEED       = 42
VAL_RATIO  = 0.15
TOPK       = 10
K_VALUES   = [5, 10, 20, 50, 100]
LR         = 0.005
EPOCHS     = 20
REG        = 0.02
REL_THRESHOLD = 4    # relevant 기준

os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. 데이터 로드 + split (★ 이 블록을 팀원과 공유 ★)
# ============================================================
df = pd.read_csv(f'{DATA_DIR}/ua.base', sep='\t',
                 names=['user', 'item', 'rating', 'timestamp'])
df['user'] -= 1   # 0-indexed (내부 계산용)
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

# 아이템 인기도 (serendipity baseline / novelty용)
popularity_counts = np.bincount(items, minlength=N_ITEMS)
popular_topk_set  = set(np.argsort(-popularity_counts)[:TOPK].tolist())


# ============================================================
# 2. 메트릭 함수
# ============================================================
def compute_beyond_accuracy(recommendations, Q, K):
    """recommendations: {user: [item0, item1, ...]}  (0-indexed)"""
    # Coverage
    rec_set = set()
    for its in recommendations.values():
        rec_set.update(its)
    coverage = len(rec_set) / N_ITEMS

    ser, nov, div = [], [], []
    for its in recommendations.values():
        topk = its[:K]
        # Serendipity: 인기 top-K에 없는 비율
        ser.append(sum(1 for i in topk if i not in popular_topk_set) / K)
        # Novelty: 평균 self-information
        novs = [-np.log2(max(popularity_counts[i] / N_USERS, 1e-9)) for i in topk]
        nov.append(np.mean(novs))
        # Diversity: 리스트 내 평균 비유사도 (Q 코사인)
        if len(topk) >= 2:
            vecs = Q[topk]
            normed = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
            sim = normed @ normed.T
            n = len(topk)
            div.append(1 - np.sum(np.triu(sim, k=1)) / (n * (n - 1) / 2))
    return coverage, np.mean(ser), np.mean(nov), np.mean(div)


# ============================================================
# 3. k별 학습 + 산출물 생성
# ============================================================
results = []
saved_k20 = {}   # grouping 분석용 (k=20 모델 보관)

for k in K_VALUES:
    print(f"\n===== latent_dim = {k} 학습 시작 =====")
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

    # augmented P (예측용)
    P_aug = P.copy()
    for u in range(N_USERS):
        if u in user_rated_items:
            I_u = user_rated_items[u]
            P_aug[u] += (1 / np.sqrt(len(I_u))) * Y[I_u].sum(axis=0)

    # --- (A) validation 예측 CSV ---
    val_pred = np.clip(mu + bu[vu] + bi[vi] + np.sum(P_aug[vu] * Q[vi], axis=1), 1, 5)
    pred_df = pd.DataFrame({
        'user_id': vu + 1,                 # 1-indexed 출력 (팀 머지용)
        'item_id': vi + 1,
        'true_rating': vr.astype(int),
        'pred_rating': np.round(val_pred, 4),
    }).sort_values(['user_id', 'item_id']).reset_index(drop=True)
    pred_df.to_csv(f'outputs/valid_mf_k{k}.csv', index=False)

    rmse = np.sqrt(np.mean((val_pred - vr) ** 2))
    mae  = np.mean(np.abs(val_pred - vr))

    # --- (B) top-10 추천 리스트 CSV ---
    train_items_per_user = {u: set(its) for u, its in user_rated_items.items()}
    recommendations = {}
    rec_rows = []
    for u in range(N_USERS):
        scores = mu + bu[u] + bi + P_aug[u] @ Q.T
        for it in train_items_per_user.get(u, set()):
            scores[it] = -np.inf            # 학습에서 본 아이템 제외
        topk = np.argpartition(-scores, TOPK)[:TOPK]
        topk = topk[np.argsort(-scores[topk])]
        recommendations[u] = topk.tolist()
        for rank, it in enumerate(topk, 1):
            rec_rows.append([u + 1, rank, it + 1,
                             round(float(np.clip(scores[it], 1, 5)), 4)])
    recs_df = pd.DataFrame(rec_rows, columns=['user_id', 'rank', 'item_id', 'pred_rating'])
    recs_df.to_csv(f'outputs/valid_mf_recs_k{k}.csv', index=False)

    # --- (C) beyond-accuracy 메트릭 ---
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

# --- 결과표 저장 ---
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/mf_results_table.csv', index=False)
print("\n========== MF 결과표 ==========")
print(results_df.to_string(index=False))
print("저장: outputs/mf_results_table.csv")


# ============================================================
# 4. Grouping 분석 (k=20 기준)
#    Low(<30) / Medium(30-80) / High(80+)  ← 학습 평점 수 기준
# ============================================================
ratings_per_user = train_set.groupby('user').size()

def user_group(u):
    n = ratings_per_user.get(u, 0)
    if n < 30:    return 'Low'
    elif n < 80:  return 'Medium'
    else:         return 'High'

# k=20 예측 로드
pred20 = pd.read_csv('outputs/valid_mf_k20.csv')
pred20['user0']   = pred20['user_id'] - 1
pred20['group']   = pred20['user0'].map(user_group)
pred20['sq_err']  = (pred20['pred_rating'] - pred20['true_rating']) ** 2
pred20['abs_err'] = (pred20['pred_rating'] - pred20['true_rating']).abs()

# k=20 추천 로드 (group별 coverage)
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
print("\n========== MF Grouping 표 (k=20) ==========")
print(group_df.to_string(index=False))
print("저장: outputs/mf_grouping_table.csv")

print("\n[완료] outputs/ 폴더에 모든 산출물 생성됨.")
