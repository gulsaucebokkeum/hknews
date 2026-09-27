"""08_descriptives.py — 부록용 사건 목록, 사건일 기술통계, 검정력 계산"""
import os, numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PROC = os.path.join(ROOT, "data", "processed"); EVD = os.path.join(ROOT, "data", "events"); TAB = os.path.join(ROOT, "output", "tables")
m = pd.read_csv(os.path.join(PROC, "master_daily.csv"), parse_dates=["date"]).set_index("date")
r = pd.read_csv(os.path.join(PROC, "regimes.csv"), parse_dates=["date"]).set_index("date")
ev = pd.read_csv(os.path.join(EVD, "events_master.csv"), parse_dates=["trade_date", "date"])
dep = ev[ev.direction.isin(["curb_depreciation", "two_sided"]) & (~ev.placeholder.astype(bool))].copy().set_index("trade_date")
lnp = m["ln_krw"]; lna = np.log(m["asia_basket"])
dep["ret_t"] = (100 * lnp.diff()).reindex(dep.index)
dep["pre5"] = (100 * (lnp.shift(1) - lnp.shift(6))).reindex(dep.index)          # t-6 → t-1
dep["post5"] = (100 * (lnp.shift(-4) - lnp.shift(1))).reindex(dep.index)        # t-1 → t+4
dep["post20"] = (100 * (lnp.shift(-19) - lnp.shift(1))).reindex(dep.index)
dep["post5_abn"] = dep["post5"] - (100 * (lna.shift(-4) - lna.shift(1))).reindex(dep.index)
dep["rv5_pre"] = m["rv5"].shift(1).reindex(dep.index); dep["rv5_post"] = m["rv5"].shift(-4).reindex(dep.index)
dep["p_stress_lag"] = r["p_filt_k3_s2"].shift(1).reindex(dep.index); dep["lvl_pct_lag"] = r["lvl_pct3y"].shift(1).reindex(dep.index)
dep["in_risk20"] = r["risk_set_20"].reindex(dep.index); dep["in_risk_nc20"] = r["risk_set_nc20"].reindex(dep.index)
dep["usdkrw"] = m["usdkrw"].reindex(dep.index)
cols = ["date", "speakers", "intensity", "joint", "official", "actual", "confidence", "usdkrw", "ret_t", "pre5", "post5", "post5_abn", "post20", "rv5_pre", "rv5_post", "p_stress_lag", "lvl_pct_lag", "in_risk20", "in_risk_nc20", "seq_5"]
out = dep.reset_index()[["trade_date"] + cols].rename(columns={"trade_date": "trade_date", "date": "stated_date"})
out.to_csv(os.path.join(TAB, "event_table_dep.csv"), index=False, float_format="%.2f")
# 기술통계: 순수 vs 실개입 동반, 공식 vs 비공식, 강도별
pure = dep[dep.actual != "Y"]
def block(g, name):
    return {"group": name, "n": len(g), "mean_ret_t": g["ret_t"].mean(), "mean_pre5": g["pre5"].mean(), "mean_post5": g["post5"].mean(), "mean_post5_abn": g["post5_abn"].mean(),
            "mean_post20": g["post20"].mean(), "share_post5_neg": (g["post5"] < 0).mean(), "mean_rv5_pre": g["rv5_pre"].mean(), "mean_rv5_post": g["rv5_post"].mean(),
            "share_in_risk20": g["in_risk20"].mean(), "share_in_risk_nc20": g["in_risk_nc20"].mean(), "share_conf_ge_med": (g["confidence"] >= 2).mean()}
rows = [block(dep, "전체(약세 억제)"), block(pure, "순수(실개입 보도 없음)"), block(dep[dep.actual == "Y"], "실개입 동반"), block(pure[pure.official], "순수·공식"), block(pure[~pure.official], "순수·비공식"),
        block(pure[pure.intensity == 1], "순수·강도1"), block(pure[pure.intensity == 2], "순수·강도2"), block(pure[pure.intensity == 3], "순수·강도3"), block(pure[pure.seq_5 == 1], "순수·단발"), block(pure[pure.seq_5 > 1], "순수·반복")]
desc = pd.DataFrame(rows); desc.to_csv(os.path.join(TAB, "event_descriptives.csv"), index=False, float_format="%.3f"); print(desc.round(2).to_string(index=False))
# 비교 기준: 위험집합 내 비사건일의 같은 통계
for rs in ["risk_set_20", "risk_set_nc20"]:
    days = r.index[(r[rs] == 1)]; days = days.difference(dep.index)
    post5 = (100 * (lnp.shift(-4) - lnp.shift(1))).reindex(days); post5a = post5 - (100 * (lna.shift(-4) - lna.shift(1))).reindex(days)
    print(f"{rs} 비사건일 n={len(days)} mean_post5={post5.mean():+.3f} mean_post5_abn={post5a.mean():+.3f} share_post5_neg={(post5<0).mean():.3f} mean_ret_t={(100*lnp.diff()).reindex(days).mean():+.3f}")
# 검정력: 이항 결과의 두 집단 비교(클러스터 미반영 근사)
def mde(n1, n0, p0, alpha=0.05, power=0.8):
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power); se = np.sqrt(p0 * (1 - p0) * (1 / n1 + 1 / n0)); return z * se
pw = []
for rs in ["risk_set_20", "risk_set_nc20", "risk_set_lvl20"]:
    days = r.index[(r[rs] == 1)]
    n1 = int(dep[dep.actual != "Y"].reindex(days).dropna(subset=["intensity"]).shape[0]); n0 = len(days) - n1
    for p0 in (0.3, 0.5): pw.append({"risk_set": rs, "n_treated": n1, "n_control": n0, "baseline_exit_prob": p0, "MDE_80pct_power": mde(n1, n0, p0)})
pw = pd.DataFrame(pw); pw.to_csv(os.path.join(TAB, "power_mde.csv"), index=False, float_format="%.3f"); print(pw.round(3).to_string(index=False))
