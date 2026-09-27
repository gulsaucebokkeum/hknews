"""
04_analysis.py — 고환율 스트레스 국면에서 구두개입의 국면이탈 효과: IPTW/AIPW, 사건연구, 벤치마크, 위약검정
==========================================================================================================
시점 규약
  - t: 거래일. 변수값은 t일 fix(서울 장 마감 이후) 기준. 처치 D_t는 t일(KST) 중 발언.
  - 사전 공변량 X_{t-1}: t-1일 fix 기준(서울 t일 개장 전에 알려진 정보).
  - 위험집합 R_t = 1{t-1일에 스트레스 국면(필터 확률>0.5) ∧ 3년 분포 상위 20%} = regimes.risk_set_20
결과변수
  - exit_h: t..t+h 사이에 HMM 스트레스 상태를 벗어나 5거래일 이상 비스트레스 유지가 시작되는지 (h=5,10,20)
  - cr_h  : 100·[ln P_{t+h-1} − ln P_{t-1}]  (h=1,5,20; +는 원화 약세)
  - dvol_h: rv_{t+h-1} − rv_{t-1}
"""
import os, json, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # 한글 폰트(NanumGothic)
except ImportError:
    pass
warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260927)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed"); EVD = os.path.join(ROOT, "data", "events")
TAB = os.path.join(ROOT, "output", "tables"); FIG = os.path.join(ROOT, "output", "figures")
FIG_ONLY = bool(os.environ.get("FIG_ONLY"))
if FIG_ONLY:
    TAB = os.path.join(ROOT, "output", "_figonly_tmp"); os.environ["B_BOOT"] = "0"; os.environ["N_PLACEBO"] = "0"
for d in (TAB, FIG): os.makedirs(d, exist_ok=True)

RISK = os.environ.get("RISK_SET", "risk_set_20")          # 강건성: risk_set_10/30/hmm/rec20/k4_20/k2_20/dir20
B_BOOT = int(os.environ.get("B_BOOT", "300")); N_PLACEBO = int(os.environ.get("N_PLACEBO", "1000"))
START = os.environ.get("START", "2000-01-03")

# ------------------------------------------------------------------ 데이터
m = pd.read_csv(os.path.join(PROC, "master_daily.csv"), parse_dates=["date"]).set_index("date")
r = pd.read_csv(os.path.join(PROC, "regimes.csv"), parse_dates=["date"]).set_index("date")
e = pd.read_csv(os.path.join(EVD, "events_daily.csv"), parse_dates=["date"]).set_index("date")
df = m.join(r.drop(columns=["usdkrw", "lvl_pct3y"]), how="left").join(e, how="left").loc[START:].copy()
for c in e.columns: df[c] = df[c].fillna(0)
STRESS_FOR = {"risk_set_20": "stress_hmm", "risk_set_10": "stress_hmm", "risk_set_30": "stress_hmm", "risk_set_hmm": "stress_hmm",
              "risk_set_rec20": "stress_rec", "risk_set_k4_20": "stress_k4", "risk_set_k2_20": "stress_k2", "risk_set_dir20": "stress_dir",
              "risk_set_nc20": "stress_nc", "risk_set_nc10": "stress_nc", "risk_set_nc30": "stress_nc", "risk_set_lvl20": "stress_nc"}
STRESS_COL = STRESS_FOR.get(RISK, "stress_hmm")
df["stress_hmm"] = df[STRESS_COL].fillna(0)     # 이탈(exit) 판정에 쓰는 당일 국면지표 = 위험집합 정의와 일치

# 결과변수
s = df["stress_hmm"].values.astype(int); n = len(s)
nonstress5 = np.array([1 if (i + 5 <= n and s[i:i + 5].sum() == 0) else (0 if i + 5 <= n else np.nan) for i in range(n)], dtype=float)
for h in (5, 10, 20):
    v = np.full(n, np.nan)
    for i in range(n):
        w = nonstress5[i:i + h + 1]
        if np.isnan(w).any(): continue
        v[i] = w.max()
    df[f"exit{h}"] = v
texit = np.full(n, np.nan)
for i in range(n):
    idx = np.where(nonstress5[i:] == 1)[0]
    texit[i] = idx[0] if len(idx) else np.nan
df["t_exit"] = texit
lnp = df["ln_krw"].values
df["cr1"] = df["ret"]
df["cr5"] = 100 * (pd.Series(lnp, index=df.index).shift(-4) - pd.Series(lnp, index=df.index).shift(1))
df["cr20"] = 100 * (pd.Series(lnp, index=df.index).shift(-19) - pd.Series(lnp, index=df.index).shift(1))
lna = np.log(df["asia_basket"].values)
df["ab1"] = df["cr1"] - df["d_asia"]                                    # 아시아 바스켓 대비 이상수익률(원화 고유 반응)
df["ab5"] = df["cr5"] - 100 * (pd.Series(lna, index=df.index).shift(-4) - pd.Series(lna, index=df.index).shift(1))
df["ab20"] = df["cr20"] - 100 * (pd.Series(lna, index=df.index).shift(-19) - pd.Series(lna, index=df.index).shift(1))
df["dvol5"] = df["rv5"].shift(-4) - df["rv5"].shift(1)
df["dvol20"] = df["rv20"].shift(-19) - df["rv20"].shift(1)

# 사전 공변량(t-1)
lagvars = ["ret", "ret5", "rv5", "lvl_pct3y", "d_dxy5", "d_asia5", "vix", "d_vix5", "d_dgs2_5", "d_cny5", "d_kospi5", "for_flow5", "p_filt_k3_s2"]
for v in lagvars: df[f"L_{v}"] = df[v].shift(1)
df["L_for_flow5"] = df["L_for_flow5"].fillna(0) / 1000.0
df["L_flow_missing"] = (df["for_flow5"].shift(1).isna()).astype(float)
df["risk"] = df[RISK].fillna(0).astype(int)
# 스트레스 스펠(위험집합 연속구간) id 및 경과일
spell = np.zeros(n, dtype=int); dur = np.zeros(n); k = 0; prev = 0
for i in range(n):
    if df["risk"].iat[i] == 1:
        if prev == 0: k += 1; dur_i = 0
        else: dur_i += 1
        spell[i] = k; dur[i] = dur_i
    prev = df["risk"].iat[i]
df["spell"] = spell; df["spell_dur"] = dur
df["L_n_dep_past20"] = df["n_dep_past20"]
for d_ in range(1, 5): df[f"dow{d_}"] = (df["dow"] == d_).astype(float)
XCOLS = [f"L_{v}" for v in lagvars] + ["L_flow_missing", "spell_dur", "L_n_dep_past20", "dow1", "dow2", "dow3", "dow4"]

OUTCOMES = ["exit5", "exit10", "exit20", "cr1", "cr5", "cr20", "ab1", "ab5", "ab20", "dvol5", "dvol20"]
TREATS = ["D_dep", "D_dep_pure", "D_dep_hq"]

# ------------------------------------------------------------------ 추정 함수
def fit_ps(X, D, C=1.0):
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    lr = LogisticRegression(C=C, max_iter=2000).fit(Xs, D)
    p = lr.predict_proba(Xs)[:, 1]
    return np.clip(p, 0.01, 0.99), lr, sc

def stab_weights(p, D):
    pD = D.mean(); w = np.where(D == 1, pD / p, (1 - pD) / (1 - p))
    lo, hi = np.percentile(w, [1, 99]); return np.clip(w, lo, hi)

def wls_rd(Y, D, w, cluster):
    X = sm.add_constant(D.astype(float))
    try:
        res = sm.WLS(Y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": cluster})
        return float(np.asarray(res.params)[1]), float(np.asarray(res.bse)[1])
    except Exception:
        return np.nan, np.nan

def aipw(Y, D, X, p):
    binary = set(np.unique(Y[~np.isnan(Y)])) <= {0.0, 1.0}
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    def fit_out(mask):
        if mask.sum() < 8: return None
        if binary and len(np.unique(Y[mask])) > 1:
            mdl = LogisticRegression(C=1.0, max_iter=2000).fit(Xs[mask], Y[mask]); return lambda Z: mdl.predict_proba(Z)[:, 1]
        elif binary:
            c = Y[mask].mean(); return lambda Z: np.full(len(Z), c)
        else:
            from sklearn.linear_model import Ridge
            mdl = Ridge(alpha=1.0).fit(Xs[mask], Y[mask]); return lambda Z: mdl.predict(Z)
    f1 = fit_out(D == 1); f0 = fit_out(D == 0)
    if f1 is None or f0 is None: return np.nan
    mu1 = f1(Xs); mu0 = f0(Xs)
    psi1 = D * (Y - mu1) / p + mu1; psi0 = (1 - D) * (Y - mu0) / (1 - p) + mu0
    return float(np.mean(psi1 - psi0))

def estimate(sub, treat, outcome, with_ps=True):
    """위험집합 표본 sub에서 outcome에 대한 naive / IPTW / AIPW 추정치"""
    d = sub[[outcome, treat, "spell"] + XCOLS].dropna()
    Y = d[outcome].values.astype(float); D = d[treat].values.astype(int); X = d[XCOLS].values
    out = {"n": len(d), "n_treated": int(D.sum()), "naive": np.nan, "iptw": np.nan, "iptw_se": np.nan, "aipw": np.nan}
    if D.sum() < 3 or (1 - D).sum() < 10: return out
    out["naive"] = float(Y[D == 1].mean() - Y[D == 0].mean())
    if with_ps:
        p, _, _ = fit_ps(X, D); w = stab_weights(p, D)
        out["iptw"], out["iptw_se"] = wls_rd(Y, D, w, d["spell"].values)
        out["aipw"] = aipw(Y, D, X, p)
    return out

BOOT_OUTCOMES = ["exit5", "exit10", "exit20", "cr5", "ab5", "dvol5"]; BOOT_TREATS = ["D_dep_pure", "D_dep_hq"]
def block_bootstrap(sub, treat, outcome, B=B_BOOT, block=20):
    """달력 기준 20거래일 이동블록 부트스트랩(전체 패널을 재표집 후 위험집합만 사용)"""
    full = df[[outcome, treat, "risk"] + XCOLS].copy(); T = len(full); est_i, est_a = [], []
    nblocks = int(np.ceil(T / block))
    for b in range(B):
        starts = rng.integers(0, T - block + 1, size=nblocks)
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in starts])[:T]
        bs = full.iloc[idx].copy(); bs["spell"] = np.repeat(np.arange(nblocks), block)[:T]   # 클러스터=블록
        bs = bs[bs["risk"] == 1]
        res = estimate(bs, treat, outcome)
        est_i.append(res["iptw"]); est_a.append(res["aipw"])
    est_i = np.array(est_i, dtype=float); est_a = np.array(est_a, dtype=float)
    q = lambda a: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)) if np.isfinite(a).sum() > 10 else (np.nan, np.nan)
    return q(est_i), q(est_a)

# ------------------------------------------------------------------ 메인 추정
sub = df[df["risk"] == 1].copy()
summary = {"risk_set": RISK, "start": START, "n_risk_days": int(len(sub)), "n_spells": int(sub["spell"].nunique()),
           "n_treated": {t: int(sub[t].sum()) for t in TREATS + ["D_dep_official", "D_dep_joint", "D_dep_hi", "D_dep_first5", "D_dep_rep5"]},
           "pilot_flag": bool(sub["D_dep_pure"].sum() < 30)}
print(json.dumps(summary, ensure_ascii=False, indent=1))

rows = []
for treat in TREATS:
    for outcome in OUTCOMES:
        res = estimate(sub, treat, outcome)
        do_boot = res["n_treated"] >= 3 and outcome in BOOT_OUTCOMES and treat in BOOT_TREATS
        (ilo, ihi), (alo, ahi) = block_bootstrap(sub, treat, outcome) if do_boot else ((np.nan, np.nan), (np.nan, np.nan))
        rows.append({"treatment": treat, "outcome": outcome, **res, "iptw_ci_lo": ilo, "iptw_ci_hi": ihi, "aipw_ci_lo": alo, "aipw_ci_hi": ahi})
        print(f"{treat:11s} {outcome:7s} n={res['n']:4d} nT={res['n_treated']:3d} naive={res['naive']:+.3f} iptw={res['iptw']:+.3f} (se {res['iptw_se']:.3f}; CI {ilo:+.3f},{ihi:+.3f}) aipw={res['aipw']:+.3f} (CI {alo:+.3f},{ahi:+.3f})")
main = pd.DataFrame(rows); main.to_csv(os.path.join(TAB, f"main_effects_{RISK}.csv"), index=False, float_format="%.4f")

# 공변량 균형(표준화 평균차) — D_dep_pure
d = sub[["D_dep_pure"] + XCOLS].dropna(); D = d["D_dep_pure"].values; X = d[XCOLS].values
if D.sum() >= 3:
    p, lr, sc = fit_ps(X, D); w = stab_weights(p, D)
    bal = []
    for j, c in enumerate(XCOLS):
        x = X[:, j]; sd = np.sqrt((x[D == 1].var() + x[D == 0].var()) / 2) + 1e-12
        smd_raw = (x[D == 1].mean() - x[D == 0].mean()) / sd
        m1 = np.average(x[D == 1], weights=w[D == 1]); m0 = np.average(x[D == 0], weights=w[D == 0])
        bal.append({"covariate": c, "mean_treated": x[D == 1].mean(), "mean_control": x[D == 0].mean(), "smd_raw": smd_raw, "smd_weighted": (m1 - m0) / sd,
                    "ps_coef_std": lr.coef_[0][j]})
    bal = pd.DataFrame(bal); bal.to_csv(os.path.join(TAB, f"balance_{RISK}.csv"), index=False, float_format="%.3f")
    print("\n균형(|SMD| 최대): raw %.2f → weighted %.2f ; PS AUC-ish 분리: p̂ 평균 처치 %.3f / 통제 %.3f" % (bal["smd_raw"].abs().max(), bal["smd_weighted"].abs().max(), p[D == 1].mean(), p[D == 0].mean()))
    summary["ps_mean_treated"] = float(p[D == 1].mean()); summary["ps_mean_control"] = float(p[D == 0].mean())
    summary["max_abs_smd_raw"] = float(bal["smd_raw"].abs().max()); summary["max_abs_smd_w"] = float(bal["smd_weighted"].abs().max())

# ------------------------------------------------------------------ 위약검정 1: 위험집합 내 무작위 날짜
plac = {}
for outcome in ["exit10", "exit20", "cr5", "ab5"]:
    d = sub[[outcome, "D_dep_pure", "spell"] + XCOLS].dropna(); nT = int(d["D_dep_pure"].sum())
    if nT < 3: continue
    actual = estimate(sub, "D_dep_pure", outcome)["iptw"]; dist = []
    if N_PLACEBO == 0: continue
    for b in range(N_PLACEBO):
        dd = d.copy(); dd["Dp"] = 0; dd.iloc[rng.choice(len(dd), nT, replace=False), dd.columns.get_loc("Dp")] = 1
        dist.append(estimate(dd, "Dp", outcome)["iptw"])
    dist = np.array(dist, dtype=float); pv = float(np.mean(np.abs(dist) >= abs(actual)))
    plac[outcome] = {"actual_iptw": actual, "placebo_mean": float(np.nanmean(dist)), "placebo_sd": float(np.nanstd(dist)), "p_value_two_sided": pv, "n_treated": nT}
    print(f"placebo {outcome}: actual={actual:+.3f} placebo mean={np.nanmean(dist):+.3f} sd={np.nanstd(dist):.3f} p={pv:.3f}")
    if outcome == "exit10":
        plt.figure(figsize=(6, 4)); plt.hist(dist[np.isfinite(dist)], bins=40, color="#9ecae1"); plt.axvline(actual, color="#d62728", lw=2, label=f"actual = {actual:+.3f}")
        plt.title("위약검정: 위험집합 내 무작위 처치일(IPTW, exit10)"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, f"fig4_placebo_{RISK}.png"), dpi=140); plt.close()
pd.DataFrame(plac).T.to_csv(os.path.join(TAB, f"placebo_random_{RISK}.csv"), float_format="%.4f")

# ------------------------------------------------------------------ 위약검정 2: 사건일 이동(±3, ±5)
shift_rows = []
for k_ in (-5, -3, 0, 3, 5):
    dfs = df.copy(); dfs["Dsh"] = dfs["D_dep_pure"].shift(k_).fillna(0)
    ss = dfs[dfs["risk"] == 1]
    for outcome in ["exit10", "cr5"]:
        res = estimate(ss, "Dsh", outcome); shift_rows.append({"shift_days": k_, "outcome": outcome, **res})
shift_tab = pd.DataFrame(shift_rows); shift_tab.to_csv(os.path.join(TAB, f"placebo_shift_{RISK}.csv"), index=False, float_format="%.4f")
print("\n사건일 이동 위약:\n", shift_tab.pivot(index="shift_days", columns="outcome", values="iptw").round(3).to_string())

# ------------------------------------------------------------------ 이질성: 강도/공동/공식/반복, 글로벌 충격 크기
het = []
for treat in ["D_dep_official", "D_dep_joint", "D_dep_hi", "D_dep_first5", "D_dep_rep5"]:
    for outcome in ["exit10", "exit20", "cr5"]:
        res = estimate(sub, treat, outcome); het.append({"treatment": treat, "outcome": outcome, **res})
# 글로벌 충격 상호작용: |d_dxy5| 중위수 기준 분할
med = sub["L_d_dxy5"].abs().median()
for lab, mask in [("small_global_shock", sub["L_d_dxy5"].abs() <= med), ("large_global_shock", sub["L_d_dxy5"].abs() > med)]:
    for outcome in ["exit10", "cr5"]:
        res = estimate(sub[mask], "D_dep_pure", outcome); het.append({"treatment": f"D_dep_pure|{lab}", "outcome": outcome, **res})
het = pd.DataFrame(het); het.to_csv(os.path.join(TAB, f"heterogeneity_{RISK}.csv"), index=False, float_format="%.4f")
print("\n이질성:\n", het[["treatment", "outcome", "n_treated", "naive", "iptw", "iptw_se", "aipw"]].round(3).to_string(index=False))

# ------------------------------------------------------------------ 반복 개입: 군집 내 2번째 이후 vs 첫 발언 (순차 IPTW)
ev = pd.read_csv(os.path.join(EVD, "events_master.csv"), parse_dates=["trade_date"])
ev = ev[ev["direction"].isin(["curb_depreciation", "two_sided"]) & (ev["actual"] != "Y")].copy()
ev = ev.set_index("trade_date").join(df[["ln_krw", "L_p_filt_k3_s2", "risk", "exit5", "exit10", "exit20", "cr5"] + XCOLS], how="left")
ev = ev[ev["risk"] == 1].copy()
rep = {}
if len(ev) >= 8 and ev["repeated_5"].sum() >= 3:
    first_lnp = ev.groupby("cluster_5")["ln_krw"].transform("first")
    ev["ret_since_first"] = 100 * (ev["ln_krw"].shift(1).fillna(ev["ln_krw"]) - first_lnp)
    ev["days_since_first"] = ev["seq_5"] - 1
    Xr = ev[["L_ret5", "L_rv5", "L_p_filt_k3_s2", "L_d_dxy5", "L_d_vix5", "ret_since_first", "days_since_first"]].fillna(0).values
    Dr = ev["repeated_5"].astype(int).values
    for outcome in ["exit5", "exit10", "exit20", "cr5"]:
        Y = ev[outcome].values.astype(float); ok = ~np.isnan(Y)
        if ok.sum() < 8 or Dr[ok].sum() < 3 or (1 - Dr[ok]).sum() < 3: continue
        p, _, _ = fit_ps(Xr[ok], Dr[ok], C=0.5); w = stab_weights(p, Dr[ok])
        rd, se = wls_rd(Y[ok], Dr[ok], w, ev["cluster_5"].values[ok])
        rep[outcome] = {"n_first": int((1 - Dr[ok]).sum()), "n_repeated": int(Dr[ok].sum()), "naive": float(Y[ok][Dr[ok] == 1].mean() - Y[ok][Dr[ok] == 0].mean()), "seq_iptw": rd, "se": se}
    print("\n반복 vs 단발(순차 IPTW):\n", pd.DataFrame(rep).T.round(3).to_string())
pd.DataFrame(rep).T.to_csv(os.path.join(TAB, f"repeated_{RISK}.csv"), float_format="%.4f")

# ------------------------------------------------------------------ 글로벌 충격 벤치마크 (위험집합 내 OLS, bp)
bench = []
Z1 = ["d_dxy", "d_dgs2", "d_vix", "d_cny"]; Z5 = ["d_dxy5", "d_dgs2_5", "d_vix5", "d_cny5"]
for horizon, ycol, zc in [("1d", "ret", Z1), ("5d", "cr5", None)]:
    d = sub.copy()
    if horizon == "5d":
        for z, base in zip(Z5, ["dxy_synth", "dgs2", "vix", "usdcny"]):
            ser = df[base]; fwd = (np.log(ser) if base != "dgs2" and base != "vix" else ser).shift(-4) - (np.log(ser) if base != "dgs2" and base != "vix" else ser).shift(1)
            d[z + "_fwd"] = (100 * fwd if base in ("dxy_synth", "usdcny") else fwd).reindex(d.index)
        zc = [z + "_fwd" for z in Z5]
    dd = d[[ycol, "D_dep_pure"] + zc].dropna()
    X = sm.add_constant(dd[zc + ["D_dep_pure"]].astype(float)); res = sm.OLS(dd[ycol].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    for z in zc:
        sd = dd[z].std(); bench.append({"horizon": horizon, "shock": z, "sd_in_stress": sd, "coef": res.params[z], "t": res.tvalues[z], "bp_per_1sd": res.params[z] * sd * 100})
    bench.append({"horizon": horizon, "shock": "D_dep_pure (구두개입일)", "sd_in_stress": np.nan, "coef": res.params["D_dep_pure"], "t": res.tvalues["D_dep_pure"], "bp_per_1sd": res.params["D_dep_pure"] * 100})
bench = pd.DataFrame(bench); bench.to_csv(os.path.join(TAB, f"benchmark_{RISK}.csv"), index=False, float_format="%.4f")
print("\n벤치마크(위험집합 내, 1sd 충격의 원/달러 반응, bp):\n", bench.round(2).to_string(index=False))

# ------------------------------------------------------------------ 그림: 사건연구 & 이탈곡선
tr = sub.index[sub["D_dep_pure"] == 1]
if len(tr) >= 3:
    H = np.arange(-5, 21); paths = []
    for t in tr:
        i = df.index.get_loc(t)
        if i - 6 < 0 or i + 20 >= n: continue
        base = lnp[i - 1]; paths.append([100 * (lnp[i + h] - base) for h in H])
    paths = np.array(paths)
    # 대조: 성향점수 최근접 5개(위험집합 내 비처치일)
    d = sub[["D_dep_pure"] + XCOLS].dropna(); p, _, _ = fit_ps(d[XCOLS].values, d["D_dep_pure"].values)
    ps = pd.Series(p, index=d.index); ctrl_paths = []
    for t in tr:
        if t not in ps.index: continue
        cand = (ps[d["D_dep_pure"] == 0] - ps[t]).abs().sort_values().index[:5]
        for c in cand:
            i = df.index.get_loc(c)
            if i - 6 < 0 or i + 20 >= n: continue
            base = lnp[i - 1]; ctrl_paths.append([100 * (lnp[i + h] - base) for h in H])
    ctrl_paths = np.array(ctrl_paths)
    mean_t = paths.mean(0); boot = np.array([paths[rng.integers(0, len(paths), len(paths))].mean(0) for _ in range(500)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    plt.figure(figsize=(8, 4.5)); plt.fill_between(H, lo, hi, color="#d62728", alpha=0.15); plt.plot(H, mean_t, color="#d62728", lw=2, label=f"구두개입일(순수, n={len(paths)})")
    if len(ctrl_paths): plt.plot(H, ctrl_paths.mean(0), color="#1f77b4", lw=2, ls="--", label=f"성향점수 대조일(n={len(ctrl_paths)})")
    plt.axhline(0, color="k", lw=0.6); plt.axvline(0, color="k", lw=0.6, ls=":"); plt.xlabel("발언일 대비 거래일"); plt.ylabel("누적 로그환율 변화(%, t-1 기준)")
    plt.title("사건연구: 고환율 스트레스 국면 내 구두개입 전후 원/달러 경로"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, f"fig2_eventstudy_{RISK}.png"), dpi=140); plt.close()
    # 이탈곡선 (가중 KM 근사): P(T_exit <= h)
    d = sub[["D_dep_pure", "t_exit", "spell"] + XCOLS].copy(); d["t_exit"] = d["t_exit"].fillna(999)
    d = d.dropna(); p, _, _ = fit_ps(d[XCOLS].values, d["D_dep_pure"].values); w = stab_weights(p, d["D_dep_pure"].values)
    hs = np.arange(0, 41); curves = {}
    for lab, mask in [("구두개입일", d["D_dep_pure"] == 1), ("비개입일", d["D_dep_pure"] == 0)]:
        te = d.loc[mask, "t_exit"].values; ww = w[mask.values]
        curves[lab] = [np.average(te <= h, weights=ww) for h in hs]
    plt.figure(figsize=(7, 4.5))
    for lab, c in curves.items(): plt.plot(hs, c, lw=2, label=lab)
    plt.xlabel("거래일 h"); plt.ylabel("P(스트레스 이탈 시작 ≤ h)  [IPTW 가중]"); plt.title("스트레스 국면 이탈의 누적확률(가중)"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"fig3_exitcurves_{RISK}.png"), dpi=140); plt.close()
    summary["exit_curve"] = {k: [round(float(x), 3) for x in v[:21]] for k, v in curves.items()}

summary["placebo"] = plac; summary["main"] = main.to_dict(orient="records")
with open(os.path.join(TAB if FIG_ONLY else os.path.join(ROOT, "output"), f"results_summary_{RISK}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
print("\ndone:", RISK)
