"""06_report.py — 결과표 통합(markdown) : output/results_summary.md"""
import os, glob, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = os.path.join(ROOT, "output", "tables")
lines = ["# 결과 요약(자동 생성)\n"]
def fmt(x, d=3):
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:+.{d}f}"
# 1. 주요 효과: 위험집합 정의별 × 처치별
files = sorted(glob.glob(os.path.join(TAB, "main_effects_*.csv")))
allm = []
for f in files:
    d = pd.read_csv(f); d["risk_set"] = os.path.basename(f).replace("main_effects_", "").replace(".csv", ""); allm.append(d)
if allm:
    M = pd.concat(allm)
    for treat in ["D_dep_pure", "D_dep_hq", "D_dep"]:
        lines.append(f"\n## 처치 = {treat}\n")
        lines.append("| 위험집합 | 결과 | n | n처치 | naive | IPTW (SE) | IPTW 95% CI | AIPW | AIPW 95% CI |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rs in ["risk_set_20", "risk_set_nc20", "risk_set_k2_20", "risk_set_lvl20", "risk_set_hmm", "risk_set_rec20", "risk_set_10", "risk_set_30", "risk_set_nc10", "risk_set_nc30"]:
            sub = M[(M.risk_set == rs) & (M.treatment == treat)]
            for _, r in sub.iterrows():
                if r["outcome"] not in ("exit5", "exit10", "exit20", "cr1", "cr5", "cr20", "ab1", "ab5", "ab20", "dvol5", "dvol20"): continue
                ci_i = f"[{fmt(r.iptw_ci_lo)}, {fmt(r.iptw_ci_hi)}]" if pd.notna(r.iptw_ci_lo) else ""
                ci_a = f"[{fmt(r.aipw_ci_lo)}, {fmt(r.aipw_ci_hi)}]" if pd.notna(r.aipw_ci_lo) else ""
                lines.append(f"| {rs} | {r.outcome} | {int(r.n)} | {int(r.n_treated)} | {fmt(r.naive)} | {fmt(r.iptw)} ({r.iptw_se:.3f}) | {ci_i} | {fmt(r.aipw)} | {ci_a} |")
# 2. 위약
for f in sorted(glob.glob(os.path.join(TAB, "placebo_random_*.csv"))):
    d = pd.read_csv(f, index_col=0); rs = os.path.basename(f).replace("placebo_random_", "").replace(".csv", "")
    if len(d) == 0: continue
    lines.append(f"\n## 위약(무작위 처치일) — {rs}\n"); lines.append("| 결과 | 실제 IPTW | 위약 평균 | 위약 SD | p(양측) | n처치 |"); lines.append("|---|---|---|---|---|---|")
    for o, r in d.iterrows(): lines.append(f"| {o} | {fmt(r.actual_iptw)} | {fmt(r.placebo_mean)} | {r.placebo_sd:.3f} | {r.p_value_two_sided:.3f} | {int(r.n_treated)} |")
for f in sorted(glob.glob(os.path.join(TAB, "placebo_shift_*.csv"))):
    d = pd.read_csv(f); rs = os.path.basename(f).replace("placebo_shift_", "").replace(".csv", "")
    if d["iptw"].notna().sum() == 0: continue
    p = d.pivot(index="shift_days", columns="outcome", values="iptw").round(3)
    lines.append(f"\n## 위약(사건일 이동, IPTW) — {rs}\n"); lines.append(p.to_markdown())
# 3. 이질성·반복·벤치마크
for f in sorted(glob.glob(os.path.join(TAB, "heterogeneity_*.csv"))):
    d = pd.read_csv(f); rs = os.path.basename(f).replace("heterogeneity_", "").replace(".csv", "")
    lines.append(f"\n## 이질성 — {rs}\n"); lines.append(d[["treatment", "outcome", "n_treated", "naive", "iptw", "iptw_se", "aipw"]].round(3).to_markdown(index=False))
for f in sorted(glob.glob(os.path.join(TAB, "repeated_*.csv"))):
    d = pd.read_csv(f, index_col=0); rs = os.path.basename(f).replace("repeated_", "").replace(".csv", "")
    if len(d) == 0: continue
    lines.append(f"\n## 반복 vs 단발(순차 IPTW) — {rs}\n"); lines.append(d.round(3).to_markdown())
for f in sorted(glob.glob(os.path.join(TAB, "benchmark_*.csv"))):
    d = pd.read_csv(f); rs = os.path.basename(f).replace("benchmark_", "").replace(".csv", "")
    lines.append(f"\n## 글로벌 충격 벤치마크(bp per 1sd) — {rs}\n"); lines.append(d.round(2).to_markdown(index=False))
for f in sorted(glob.glob(os.path.join(TAB, "tvtp_results_lag*.csv"))):
    d = pd.read_csv(f); lines.append(f"\n## TVTP-MS — {os.path.basename(f)}\n"); lines.append(d.round(3).to_markdown(index=False))
for f in sorted(glob.glob(os.path.join(TAB, "balance_*.csv"))):
    d = pd.read_csv(f); rs = os.path.basename(f).replace("balance_", "").replace(".csv", "")
    lines.append(f"\n## 공변량 균형(D_dep_pure) — {rs}\n"); lines.append(d.round(3).to_markdown(index=False))
open(os.path.join(ROOT, "output", "results_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
print("written", len(lines), "lines")
