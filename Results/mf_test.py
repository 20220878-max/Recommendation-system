# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

"""
mf_test.py — SVD++ 최종 평가 스크립트
CSV로 저장된 4단계 모델을 불러와서 ua.test로 최종 RMSE/MAE 측정
교수님 요구사항: train/test 코드 분리, ua.test는 여기서만 1번 사용
"""
import os
import numpy as np
import pandas as pd

# ============================================
# 설정
# ============================================
DATA_DIR = 'raw/ml-100k'
N_USERS = 943
N_ITEMS = 1682
K = 50

MODEL_DIR = 'model'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("SVD++ 최종 평가 — ua.test (4 Stages 비교)")
print("=" * 65)


def load_model_csv(stage_dir):
    """CSV로 저장된 모델 파라미터 로드"""
    model = {}
    for f in os.listdir(stage_dir):
        if f.endswith('.csv'):
            name = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(stage_dir, f))
            if len(df.columns) == 1:
                model[name] = df.iloc[:, 0].values
            else:
                model[name] = df.values
    return model


# ============================================
# 1. ua.base 로드 (P_aug 계산용, 학습 아님!)
# ============================================
train_df = pd.read_csv(f'{DATA_DIR}/ua.base', sep='\t',
                       names=['user', 'item', 'rating', 'timestamp'])
train_df['user'] -= 1
train_df['item'] -= 1
user_rated_items = {u: g['item'].values for u, g in train_df.groupby('user')}
ratings_per_user = train_df.groupby('user').size()
mu_global = train_df['rating'].mean()
print(f"  ua.base: {len(train_df)}개 평점 (P_aug 계산용)")

# ============================================
# 2. ua.test 로드
# ============================================
test_df = pd.read_csv(f'{DATA_DIR}/ua.test', sep='\t',
                      names=['user', 'item', 'rating', 'timestamp'])
test_df['user'] -= 1
test_df['item'] -= 1

tu = test_df['user'].values
ti = test_df['item'].values
tr = test_df['rating'].values.astype(float)
print(f"  ua.test: {len(test_df)}개 평점\n")

# ============================================
# 3. 각 Stage 모델 로드 + 예측 + 평가
# ============================================
results = []

# --- Baseline ---
baseline_pred = np.full_like(tr, mu_global)
baseline_rmse = np.sqrt(np.mean((baseline_pred - tr) ** 2))
baseline_mae  = np.mean(np.abs(baseline_pred - tr))
results.append(('Baseline', '전체 평균', baseline_rmse, baseline_mae, baseline_pred))

# --- Stage 1: Basic MF ---
stage1_dir = f'{MODEL_DIR}/stage1_k{K}'
if os.path.exists(stage1_dir):
    m1 = load_model_csv(stage1_dir)
    mu1 = m1['mu'][0]
    pred1 = np.clip(mu1 + np.sum(m1['P'][tu] * m1['Q'][ti], axis=1), 1, 5)
    rmse1 = np.sqrt(np.mean((pred1 - tr) ** 2))
    mae1  = np.mean(np.abs(pred1 - tr))
    results.append(('Stage 1', 'Basic MF (P·Q)', rmse1, mae1, pred1))
    print(f"  성공 Stage 1 로드: {stage1_dir}/")
else:
    print(f"  실패 Stage 1 없음: {stage1_dir}/")

# --- Stage 2: + L2 ---
stage2_dir = f'{MODEL_DIR}/stage2_k{K}'
if os.path.exists(stage2_dir):
    m2 = load_model_csv(stage2_dir)
    mu2 = m2['mu'][0]
    pred2 = np.clip(mu2 + np.sum(m2['P'][tu] * m2['Q'][ti], axis=1), 1, 5)
    rmse2 = np.sqrt(np.mean((pred2 - tr) ** 2))
    mae2  = np.mean(np.abs(pred2 - tr))
    results.append(('Stage 2', '+ L2 정규화', rmse2, mae2, pred2))
    print(f"  성공 Stage 2 로드: {stage2_dir}/")
else:
    print(f"  실패 Stage 2 없음: {stage2_dir}/")

# --- Stage 3: + Bias ---
stage3_dir = f'{MODEL_DIR}/stage3_k{K}'
if os.path.exists(stage3_dir):
    m3 = load_model_csv(stage3_dir)
    mu3 = m3['mu'][0]
    pred3 = np.clip(mu3 + m3['bu'][tu] + m3['bi'][ti] + np.sum(m3['P'][tu] * m3['Q'][ti], axis=1), 1, 5)
    rmse3 = np.sqrt(np.mean((pred3 - tr) ** 2))
    mae3  = np.mean(np.abs(pred3 - tr))
    results.append(('Stage 3', '+ Bias (mu,bu,bi)', rmse3, mae3, pred3))
    print(f"  성공 Stage 3 로드: {stage3_dir}/")
else:
    print(f"  실패 Stage 3 없음: {stage3_dir}/")

# --- Stage 4: + Implicit (SVD++) ---
stage4_dir = f'{MODEL_DIR}/stage4_k{K}'
if os.path.exists(stage4_dir):
    m4 = load_model_csv(stage4_dir)
    mu4 = m4['mu'][0]
    P4, Q4, Y4, bu4, bi4 = m4['P'], m4['Q'], m4['Y'], m4['bu'], m4['bi']

    P4_aug = P4.copy()
    for u in range(N_USERS):
        if u in user_rated_items:
            I_u = user_rated_items[u]
            P4_aug[u] += (1 / np.sqrt(len(I_u))) * Y4[I_u].sum(axis=0)

    pred4 = np.clip(mu4 + bu4[tu] + bi4[ti] + np.sum(P4_aug[tu] * Q4[ti], axis=1), 1, 5)
    rmse4 = np.sqrt(np.mean((pred4 - tr) ** 2))
    mae4  = np.mean(np.abs(pred4 - tr))
    results.append(('Stage 4', '+ Implicit (SVD++)', rmse4, mae4, pred4))
    print(f"  성공 Stage 4 로드: {stage4_dir}/")
else:
    print(f"  실패 Stage 4 없음: {stage4_dir}/")

# ============================================
# 4. 단계별 결과 비교표
# ============================================
print(f"\n{'=' * 65}")
print(" ua.test 최종 결과 — 단계별 비교")
print(f"{'=' * 65}")
print(f"  {'Stage':>10} {'내용':>25} {'RMSE':>8} {'MAE':>8} {'RMSE 개선':>10}")
print("  " + "-" * 63)

prev_rmse = None
for stage, desc, rmse, mae, _ in results:
    improve = '—' if prev_rmse is None else f"{prev_rmse - rmse:>+.4f}"
    print(f"  {stage:>10} {desc:>25} {rmse:>8.4f} {mae:>8.4f} {improve:>10}")
    prev_rmse = rmse

if len(results) >= 5:
    total_improve = results[0][2] - results[-1][2]
    pct = total_improve / results[0][2] * 100
    print(f"\n  총 개선: Baseline({results[0][2]:.4f}) → Stage 4({results[-1][2]:.4f}) = {total_improve:.4f} ({pct:.1f}%)")

# ============================================
# 5. Stage 4 Grouping 분석
# ============================================
if len(results) >= 5:
    final_pred = results[-1][4]

    result_df = pd.DataFrame({
        'user_id': tu + 1, 'item_id': ti + 1,
        'true_rating': tr.astype(int),
        'pred_rating': np.round(final_pred, 4),
    }).sort_values(['user_id', 'item_id']).reset_index(drop=True)

    def user_group(u):
        n = ratings_per_user.get(u, 0)
        if n < 30:    return 'Cold (<30)'
        elif n < 80:  return 'Medium (30-80)'
        else:         return 'Active (80+)'

    result_df['user0'] = result_df['user_id'] - 1
    result_df['group'] = result_df['user0'].map(user_group)
    result_df['sq_err'] = (result_df['pred_rating'] - result_df['true_rating']) ** 2
    result_df['abs_err'] = (result_df['pred_rating'] - result_df['true_rating']).abs()

    print(f"\n{'=' * 65}")
    print(" Stage 4 (SVD++) Grouping 분석 — ua.test")
    print(f"{'=' * 65}")
    print(f"  {'그룹':>18} {'사용자수':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 45)

    for grp in ['Cold (<30)', 'Medium (30-80)', 'Active (80+)']:
        g = result_df[result_df['group'] == grp]
        if len(g) == 0:
            continue
        print(f"  {grp:>18} {g['user_id'].nunique():>8} {np.sqrt(g['sq_err'].mean()):>8.4f} {g['abs_err'].mean():>8.4f}")

    print(f"  {'전체':>18} {result_df['user_id'].nunique():>8} {results[-1][2]:>8.4f} {results[-1][3]:>8.4f}")

    # 결과 저장
    result_path = f'{OUTPUT_DIR}/test_mf_k{K}_final.csv'
    result_df[['user_id', 'item_id', 'true_rating', 'pred_rating']].to_csv(result_path, index=False)

    summary_rows = [{'stage': s, 'description': d, 'data': 'ua.test',
                     'RMSE': round(r, 4), 'MAE': round(m, 4)}
                    for s, d, r, m, _ in results]
    pd.DataFrame(summary_rows).to_csv(f'{OUTPUT_DIR}/test_mf_stages_summary.csv', index=False)

    print(f"\n  저장: {result_path}")
    print(f"  저장: {OUTPUT_DIR}/test_mf_stages_summary.csv")

print(f"\n{'=' * 65}")
print("최종 평가 완료!")
print(f"{'=' * 65}")
