"""
mf_train.py — SVD++ 단계적 학습 스크립트
Baseline → Stage 1 (Basic MF) → Stage 2 (+L2) → Stage 3 (+Bias) → Stage 4 (+Implicit/SVD++)
ua.base 전체(100%)로 학습 후 각 Stage 모델을 CSV로 저장
"""
import os
import time
import numpy as np
import pandas as pd

# ============================================
# 하이퍼파라미터
# ============================================
DATA_DIR = 'ml-100k'
N_USERS = 943
N_ITEMS = 1682
SEED = 42
K = 50          # best latent_dim (RMSE 0.9192)
LR = 0.005
EPOCHS = 20
REG = 0.02

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# 데이터 로드 (ua.base 전체, split 없음)
# ============================================
print("=" * 65)
print("SVD++ 단계적 학습 (4 Stages)")
print(f"  latent_dim={K}, lr={LR}, reg={REG}, epochs={EPOCHS}")
print("=" * 65)

df = pd.read_csv(f'{DATA_DIR}/ua.base', sep='\t',
                 names=['user', 'item', 'rating', 'timestamp'])
df['user'] -= 1
df['item'] -= 1

users   = df['user'].values
items   = df['item'].values
ratings = df['rating'].values.astype(float)
user_rated_items = {u: g['item'].values for u, g in df.groupby('user')}

mu = ratings.mean()
print(f"  학습 데이터: {len(df)}개 평점 (ua.base 전체)")
print(f"  global mean (μ): {mu:.4f}\n")


def compute_train_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))


def save_model_csv(stage_dir, **arrays):
    """모델 파라미터를 CSV로 저장 (VS Code에서 확인 가능)"""
    os.makedirs(stage_dir, exist_ok=True)
    for name, arr in arrays.items():
        path = os.path.join(stage_dir, f'{name}.csv')
        if arr.ndim == 1:
            pd.DataFrame({name: arr}).to_csv(path, index=False)
        elif arr.ndim == 2:
            col_names = [f'dim_{i}' for i in range(arr.shape[1])]
            pd.DataFrame(arr, columns=col_names).to_csv(path, index=False)
        else:
            pd.DataFrame({name: [arr.item()]}).to_csv(path, index=False)
    print(f"  모델 저장: {stage_dir}/")


# ============================================
# Baseline: 전체 평균 예측
# ============================================
print("-" * 65)
print("[Baseline] 전체 평균 예측")
baseline_rmse = compute_train_rmse(np.full_like(ratings, mu), ratings)
print(f"  Train RMSE: {baseline_rmse:.4f}")


# ============================================
# Stage 1: Basic MF (P·Q만, 정규화 없음)
# ============================================
print("\n" + "-" * 65)
print("[Stage 1] Basic MF — r = mu + P[u]·Q[i], 정규화 없음")
start = time.time()

np.random.seed(SEED)
P1 = np.random.normal(0, 0.1, (N_USERS, K))
Q1 = np.random.normal(0, 0.1, (N_ITEMS, K))

for epoch in range(EPOCHS):
    order = np.random.permutation(len(ratings))
    for j in order:
        u, i, r = users[j], items[j], ratings[j]
        err = r - (mu + P1[u] @ Q1[i])
        P1[u] += LR * (err * Q1[i])
        Q1[i] += LR * (err * P1[u])

pred1 = mu + np.sum(P1[users] * Q1[items], axis=1)
s1_rmse = compute_train_rmse(np.clip(pred1, 1, 5), ratings)
s1_time = time.time() - start
print(f"  Train RMSE: {s1_rmse:.4f}  ({s1_time:.0f}초)")

save_model_csv(f'{MODEL_DIR}/stage1_k{K}',
               P=P1, Q=Q1, mu=np.array([mu]))


# ============================================
# Stage 2: + L2 정규화
# ============================================
print("\n" + "-" * 65)
print(f"[Stage 2] + L2 정규화 (reg={REG}) — 과적합 방지")
start = time.time()

np.random.seed(SEED)
P2 = np.random.normal(0, 0.1, (N_USERS, K))
Q2 = np.random.normal(0, 0.1, (N_ITEMS, K))

for epoch in range(EPOCHS):
    order = np.random.permutation(len(ratings))
    for j in order:
        u, i, r = users[j], items[j], ratings[j]
        err = r - (mu + P2[u] @ Q2[i])
        P2[u] += LR * (err * Q2[i] - REG * P2[u])
        Q2[i] += LR * (err * P2[u] - REG * Q2[i])

pred2 = mu + np.sum(P2[users] * Q2[items], axis=1)
s2_rmse = compute_train_rmse(np.clip(pred2, 1, 5), ratings)
s2_time = time.time() - start
print(f"  Train RMSE: {s2_rmse:.4f}  ({s2_time:.0f}초)")

save_model_csv(f'{MODEL_DIR}/stage2_k{K}',
               P=P2, Q=Q2, mu=np.array([mu]))


# ============================================
# Stage 3: + Bias (mu + bu + bi)
# ============================================
print("\n" + "-" * 65)
print("[Stage 3] + Bias terms (mu, bu, bi) — 사용자/아이템 편향 모델링")
start = time.time()

np.random.seed(SEED)
P3  = np.random.normal(0, 0.1, (N_USERS, K))
Q3  = np.random.normal(0, 0.1, (N_ITEMS, K))
bu3 = np.zeros(N_USERS)
bi3 = np.zeros(N_ITEMS)

for epoch in range(EPOCHS):
    order = np.random.permutation(len(ratings))
    for j in order:
        u, i, r = users[j], items[j], ratings[j]
        err = r - (mu + bu3[u] + bi3[i] + P3[u] @ Q3[i])
        bu3[u] += LR * (err - REG * bu3[u])
        bi3[i] += LR * (err - REG * bi3[i])
        P3[u]  += LR * (err * Q3[i] - REG * P3[u])
        Q3[i]  += LR * (err * P3[u] - REG * Q3[i])

pred3 = mu + bu3[users] + bi3[items] + np.sum(P3[users] * Q3[items], axis=1)
s3_rmse = compute_train_rmse(np.clip(pred3, 1, 5), ratings)
s3_time = time.time() - start
print(f"  Train RMSE: {s3_rmse:.4f}  ({s3_time:.0f}초)")

save_model_csv(f'{MODEL_DIR}/stage3_k{K}',
               P=P3, Q=Q3, bu=bu3, bi=bi3, mu=np.array([mu]))


# ============================================
# Stage 4: + Implicit Feedback (SVD++)
# ============================================
print("\n" + "-" * 65)
print("[Stage 4] + Implicit Feedback (SVD++) — 평가 이력 자체를 신호로 활용")
start = time.time()

np.random.seed(SEED)
P4  = np.random.normal(0, 0.1, (N_USERS, K))
Q4  = np.random.normal(0, 0.1, (N_ITEMS, K))
Y4  = np.random.normal(0, 0.1, (N_ITEMS, K))
bu4 = np.zeros(N_USERS)
bi4 = np.zeros(N_ITEMS)

for epoch in range(EPOCHS):
    order = np.random.permutation(len(ratings))
    for j in order:
        u, i, r = users[j], items[j], ratings[j]
        I_u = user_rated_items[u]
        norm = 1.0 / np.sqrt(len(I_u))
        p_aug = P4[u] + norm * Y4[I_u].sum(axis=0)
        err = r - (mu + bu4[u] + bi4[i] + p_aug @ Q4[i])
        bu4[u]  += LR * (err - REG * bu4[u])
        bi4[i]  += LR * (err - REG * bi4[i])
        P4[u]   += LR * (err * Q4[i] - REG * P4[u])
        Q4[i]   += LR * (err * p_aug - REG * Q4[i])
        Y4[I_u] += LR * (err * norm * Q4[i] - REG * Y4[I_u])

pred4 = mu + bu4[users] + bi4[items]
P4_aug = P4.copy()
for u in range(N_USERS):
    if u in user_rated_items:
        I_u = user_rated_items[u]
        P4_aug[u] += (1 / np.sqrt(len(I_u))) * Y4[I_u].sum(axis=0)
pred4 = mu + bu4[users] + bi4[items] + np.sum(P4_aug[users] * Q4[items], axis=1)
s4_rmse = compute_train_rmse(np.clip(pred4, 1, 5), ratings)
s4_time = time.time() - start
print(f"  Train RMSE: {s4_rmse:.4f}  ({s4_time:.0f}초)")

save_model_csv(f'{MODEL_DIR}/stage4_k{K}',
               P=P4, Q=Q4, Y=Y4, bu=bu4, bi=bi4, mu=np.array([mu]))


# ============================================
# 결과 요약
# ============================================
print("\n" + "=" * 65)
print("단계별 Train RMSE 요약")
print("=" * 65)
print(f"  {'Stage':>10} {'내용':>30} {'RMSE':>8} {'개선':>10}")
print("  " + "-" * 60)
print(f"  {'Baseline':>10} {'전체 평균':>30} {baseline_rmse:>8.4f} {'—':>10}")
print(f"  {'Stage 1':>10} {'Basic MF (P·Q)':>30} {s1_rmse:>8.4f} {baseline_rmse - s1_rmse:>+10.4f}")
print(f"  {'Stage 2':>10} {'+ L2 정규화':>30} {s2_rmse:>8.4f} {s1_rmse - s2_rmse:>+10.4f}")
print(f"  {'Stage 3':>10} {'+ Bias (mu,bu,bi)':>30} {s3_rmse:>8.4f} {s2_rmse - s3_rmse:>+10.4f}")
print(f"  {'Stage 4':>10} {'+ Implicit (SVD++)':>30} {s4_rmse:>8.4f} {s3_rmse - s4_rmse:>+10.4f}")

total_time = s1_time + s2_time + s3_time + s4_time
print(f"\n  총 학습 시간: {total_time:.0f}초")
print(f"  모델 저장:")
for s in [1,2,3,4]:
    print(f"    model/stage{s}_k{K}/  (P.csv, Q.csv, ...)")
print(f"\n{'=' * 65}")
print("학습 완료! 다음: python mf_test.py")
print(f"{'=' * 65}")
