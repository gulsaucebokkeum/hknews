"""07_figures_case.py — 사례연구 그림: 2022년, 2025-26년 스트레스 국면과 구두개입"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # 한글 폰트(NanumGothic)
except ImportError:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PROC = os.path.join(ROOT, "data", "processed"); EVD = os.path.join(ROOT, "data", "events"); FIG = os.path.join(ROOT, "output", "figures")
m = pd.read_csv(os.path.join(PROC, "master_daily.csv"), parse_dates=["date"]).set_index("date")
r = pd.read_csv(os.path.join(PROC, "regimes.csv"), parse_dates=["date"]).set_index("date")
ev = pd.read_csv(os.path.join(EVD, "events_master.csv"), parse_dates=["trade_date"])
dep = ev[ev.direction.isin(["curb_depreciation", "two_sided"]) & (~ev.placeholder.astype(bool))]
def panel(ax, a, b, title):
    x = m.loc[a:b]; rr = r.loc[a:b]
    ax.plot(x.index, x["usdkrw"], color="#1f4e79", lw=1.2)
    ax.fill_between(rr.index, x["usdkrw"].min(), x["usdkrw"].max(), where=(rr["stress_20"] == 1), color="#d62728", alpha=0.15, lw=0, label="스트레스(최상위∧상위20%)")
    ax.fill_between(rr.index, x["usdkrw"].min(), x["usdkrw"].max(), where=((rr["stress_nc"] == 1) & (rr["lvl_pct3y"] >= 0.8) & (rr["stress_20"] == 0)), color="#ff7f0e", alpha=0.12, lw=0, label="비평온∧상위20%(추가)")
    e = dep[(dep.trade_date >= a) & (dep.trade_date <= b)]
    for _, row in e.iterrows():
        if row.trade_date not in x.index: continue
        y = x.loc[row.trade_date, "usdkrw"]; c = "#d62728" if row.actual != "Y" else "#000000"; mk = "^" if row.official else "o"
        ax.scatter([row.trade_date], [y], marker=mk, s=60 if row.intensity >= 2 else 30, color=c, zorder=5, edgecolor="k", lw=0.5)
    ax.set_title(title); ax.set_ylabel("USD/KRW")
fig, ax = plt.subplots(3, 1, figsize=(13, 11))
panel(ax[0], "2008-03-01", "2009-06-30", "2008–09 글로벌 금융위기: ▲ 공식 발언, ● 비공식, 검정 = 당일 실개입 보도")
panel(ax[1], "2022-01-01", "2023-01-31", "2022 연준 긴축기")
panel(ax[2], "2025-09-01", "2026-09-18", "2025–26 원화 약세기")
ax[0].legend(loc="upper left", fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig7_case_episodes.png"), dpi=140); plt.close(); print("saved fig7")
