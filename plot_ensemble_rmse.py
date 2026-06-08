# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ENS_DIR = Path("ensemble_outputs")

# ── 데이터 ──────────────────────────────────────────────────────────────
sweep_df = pd.read_csv(ENS_DIR / "stage6_all_pair_weight_sweep.csv")
sweep_line = sweep_df[(sweep_df["usercf_k"] == 20) & (sweep_df["mf_k"] == 50)].sort_values("w_usercf")

grouping_df = pd.read_csv(ENS_DIR / "stage6_grouping_summary.csv")

# ── 색상 ─────────────────────────────────────────────────────────────────
C_BLUE   = "#4472C4"
C_ORANGE = "#ED7D31"
C_PINK   = "#E07070"
C_YELLOW = "#FFC000"
C_GREEN  = "#70AD47"

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("앙상블 분석 결과", fontsize=16, fontweight="bold", y=0.98)

# ── [좌상] 단계별 RMSE 변화 (순서: UserCF → Bagging → Weighted → Boosting) ──
ax = axes[0, 0]
labels  = ["UserCF\n단독", "Bagging\nk=50", "Weighted\nEnsemble", "Boosting\nk=50"]
rmse_v  = [1.0156,          1.0208,           0.9182,               1.0175]
colors  = [C_BLUE,          C_PINK,           C_GREEN,              C_PINK]

bars = ax.bar(labels, rmse_v, color=colors, width=0.5, edgecolor="white", zorder=3)
for bar, val in zip(bars, rmse_v):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.axhline(y=1.0156, color="gray", linestyle="--", linewidth=1.2, label="UserCF 기준선", zorder=2)
ax.set_ylim(0.88, 1.065)
ax.set_ylabel("RMSE (낮을수록 좋음)", fontsize=9)
ax.set_title("단계별 RMSE 변화", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=1)

# ── [우상] MF k=50 기준 가중치 sweep ─────────────────────────────────────
ax = axes[0, 1]
ax.plot(sweep_line["w_usercf"], sweep_line["RMSE"], color=C_BLUE, marker="o", markersize=5, linewidth=2)
ax.axvline(x=0.1, color="red", linestyle="--", linewidth=1.5, label="최적 w_usercf=0.1")
ax.set_xlabel("UserCF 가중치 (w_usercf)", fontsize=9)
ax.set_ylabel("RMSE", fontsize=9)
ax.set_title("MF k=50 기준 가중치 sweep", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.3)

# ── [좌하] 그룹별 RMSE 비교 ───────────────────────────────────────────────
ax = axes[1, 0]
models_g = ["UserCF_best_k50", "MF_best_k50", "Weighted_all_pair_best"]
labels_g = ["UserCF k=50", "MF k=50", "W.Ens"]
colors_g = [C_BLUE, C_ORANGE, C_GREEN]
groups   = ["cold", "medium", "active"]
xlabels  = ["Cold\n(평점≤32)", "Medium\n(32~104)", "Active\n(104+)"]

x = np.arange(len(groups))
w = 0.25
for i, (m, lbl, c) in enumerate(zip(models_g, labels_g, colors_g)):
    vals = [grouping_df[(grouping_df["model"] == m) & (grouping_df["group"] == g)]["RMSE"].values[0]
            for g in groups]
    ax.bar(x + i * w, vals, width=w, color=c, label=lbl, edgecolor="white")

ax.set_xticks(x + w)
ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel("RMSE", fontsize=9)
ax.set_ylim(0.85, 1.10)
ax.set_title("그룹별 RMSE 비교", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# ── [우하] 전체 메트릭 비교 ───────────────────────────────────────────────
ax = axes[1, 1]
metrics_map = {
    "UserCF":   {"RMSE": 1.0156, "Coverage": 0.4810, "Serendipity": 0.9122, "Diversity": 0.7255},
    "MF":       {"RMSE": 0.9192, "Coverage": 0.5059, "Serendipity": 0.9142, "Diversity": 0.7252},
    "Bagging":  {"RMSE": 1.0208, "Coverage": 0.4869, "Serendipity": 0.9144, "Diversity": 0.7249},
    "Boosting": {"RMSE": 1.0175, "Coverage": 0.4893, "Serendipity": 0.9129, "Diversity": 0.7255},
    "W.Ens":    {"RMSE": 0.9182, "Coverage": 0.5012, "Serendipity": 0.9138, "Diversity": 0.7249},
}
met_labels = ["RMSE", "Coverage", "Serendipity", "Diversity"]
model_names = list(metrics_map.keys())
model_colors = [C_BLUE, C_ORANGE, C_PINK, C_YELLOW, C_GREEN]

x2 = np.arange(len(met_labels))
w2 = 0.15
for i, (name, c) in enumerate(zip(model_names, model_colors)):
    vals = [metrics_map[name][m] for m in met_labels]
    ax.bar(x2 + i * w2, vals, width=w2, color=c, label=name, edgecolor="white")

ax.set_xticks(x2 + w2 * 2)
ax.set_xticklabels(met_labels, fontsize=9)
ax.set_ylabel("값", fontsize=9)
ax.set_title("전체 메트릭 비교", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# ── 저장 ──────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("ensemble_ppt_graphs.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"저장 완료: {out.resolve()}")
