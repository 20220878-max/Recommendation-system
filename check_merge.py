Stage 1 결과 정리---------------------------------------------------------------------------------------------------------------------
MF 단독 성능
k	RMSE	MAE
50	0.9192	0.7232
20	0.9196	0.7239
10	0.9236	0.7274
100	0.9239	0.7289
5	0.9257	0.7293

MF는 k=50이 가장 좋음.

UserCF 단독 성능
k	RMSE	MAE
50	1.0156	0.8053
100	1.0177	0.8091
20	1.0244	0.8111
10	1.0450	0.8263
5	1.0870	0.8571

UserCF도 k=50이 가장 좋음.

여기서 해석

현재 validation 기준으로는 MF가 UserCF보다 정확도 지표에서 더 좋다고 볼 수 있어.

MF best RMSE    : 0.9192
UserCF best RMSE: 1.0156

즉, 앞으로 앙상블에서는 보통 MF 비중이 더 높게 들어갈 가능성이 커.

예상되는 weighted ensemble 최적 방향은 이런 식일 수 있어.

UserCF 10~30%
MF 70~90%

물론 실제로는 Stage 3에서 계산해봐야 해.

Stage 1 보고서/노션 문장

이렇게 쓰면 돼.

Stage 1에서는 팀원들이 구현한 UserCF와 MF 모델의 validation 예측 결과를 동일한 기준으로 정리하였다. 두 모델 모두 ua.base를 85:15로 분리한 동일 validation set에 대해 예측했으며, check_merge.py를 통해 모든 k에서 13,586개의 validation row가 완전히 일치함을 확인하였다. 단일 모델 성능 비교 결과, UserCF와 MF 모두 k=50에서 가장 낮은 RMSE를 보였으며, 정확도 기준으로는 MF가 UserCF보다 우수한 성능을 보였다.

생성된 파일

이 파일이 Stage 1 결과물이야.

ensemble_outputs/stage1_baseline_summary.csv

Stage 2 결과 해석------------------------------------------------------------------------------------------------------------------

Bagging 결과는 이렇게야.

model	k	n_bags	RMSE	MAE
TrueBagging_UserCF	50	5	1.0208	0.8117
TrueBagging_UserCF	100	5	1.0213	0.8141
TrueBagging_UserCF	20	5	1.0269	0.8154
TrueBagging_UserCF	10	5	1.0372	0.8229
TrueBagging_UserCF	5	5	1.0600	0.8406

즉, True Bagging UserCF의 best k도 k=50이야.

중요한 분석 포인트

Stage 1의 UserCF 단독 best는:

UserCF k=50
RMSE = 1.0156
MAE  = 0.8053

Stage 2의 True Bagging UserCF best는:

TrueBagging UserCF k=50
RMSE = 1.0208
MAE  = 0.8117

즉, 이번 bagging은 정확도 기준으로는 단일 UserCF보다 약간 나빠졌어.

근데 이건 실패가 아니라 분석 포인트야. 교수님이 보는 건 “무조건 좋아졌냐”만이 아니라, 시도했고, 왜 결과가 이렇게 나왔는지 분석했는가야.

보고서에는 이렇게 쓰면 좋아.

Stage 2에서는 UserCF에 bootstrap bagging을 적용하였다. ua.base를 85:15로 분리한 뒤, train set에서 복원추출 방식으로 5개의 bootstrap sample을 생성하고 각 sample에 대해 UserCF를 독립적으로 학습하였다. 이후 동일 validation set에 대한 5개 모델의 예측값을 평균내어 최종 예측값을 생성하였다. 실험 결과 k=50에서 가장 낮은 RMSE를 보였으나, 단일 UserCF best 결과보다 RMSE가 소폭 증가하였다. 이는 bootstrap sampling 과정에서 일부 사용자-아이템 평점 정보가 반복되거나 누락되어, 희소한 평점 행렬에서 이웃 기반 CF의 유사도 계산 안정성이 오히려 낮아졌기 때문으로 해석할 수 있다.

이 문장이 진짜 중요해.
“Bagging 했는데 안 좋아짐” → 감점이 아니라,
“Bagging 했고 왜 성능이 안 좋아졌는지 분석함” → 시간 투자/분석 점수 방어야.

현재 진행 상태
완료:
Stage 1. UserCF / MF 단일 모델 결과 정리
Stage 2. True Bagging UserCF 구현 및 결과 저장

다음:
Stage 3. Weighted Ensemble 구현

Stage 3에서는 이제 팀원들이 준 UserCF/MF CSV를 다시 사용해서:

pred = w_user * UserCF_pred + (1 - w_user) * MF_pred

를 계산할 거야.

그리고 w_user를 이렇게 바꿔.

0.0, 0.1, 0.2, ..., 1.0

Stage 1 결과상 MF가 UserCF보다 훨씬 좋았기 때문에, 아마 최적 weight는 MF 비중이 큰 쪽으로 나올 가능성이 높아.