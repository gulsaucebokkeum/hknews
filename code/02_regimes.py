"""
02_regimes.py — 은닉 마르코프(Markov-switching) 국면 식별
--------------------------------------------------------
- 기본 사양: 3상태 가우시안 HMM, 관측 = [일별 로그수익률(%), ln(5일 실현변동성)]
- 상태는 수익률 변동성 오름차순으로 정렬해 0=평온, 1=변동성 확대, 2=스트레스로 명명
- filtered 확률은 적합된 모수로 forward 필터를 직접 계산(전일 정보만 사용)
- 강건성: 2·4상태, 방향성 특징(ret20) 추가 사양, 연도별 확장창(recursive) 재추정
출력: data/processed/regimes.csv, output/tables/hmm_params_*.csv, output/figures/fig1_regimes.png
"""
import os, warnings
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # 한글 폰트(NanumGothic)
except ImportError:
    pass
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed"); TAB = os.path.join(ROOT, "output", "tables"); FIG = os.path.join(ROOT, "output", "figures")
for d in (TAB, FIG): os.makedirs(d, exist_ok=True)

m = pd.read_csv(os.path.join(PROC, "master_daily.csv"), parse_dates=["date"]).set_index("date")
m = m.loc["2000-01-03":].copy()
m["ln_rv5"] = np.log(m["rv5"])

def fit_best(X, k, seeds=range(6)):
    best = None
    for s in seeds:
        h = GaussianHMM(n_components=k, covariance_type="full", n_iter=2000, tol=1e-6, random_state=s)
        try:
            h.fit(X)
            ll = h.score(X)
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, h)
    return best

def frame_loglik(h, X):
    return np.column_stack([multivariate_normal.logpdf(X, mean=h.means_[j], cov=h.covars_[j], allow_singular=True)
                            for j in range(h.n_components)])

def forward_filter(h, X):
    """filtered P(s_t | y_1..t) 와 one-step predicted P(s_t | y_1..t-1)"""
    B = frame_loglik(h, X); A = h.transmat_; T, K = B.shape
    F = np.zeros((T, K)); P = np.zeros((T, K)); pred = h.startprob_.copy()
    for t in range(T):
        P[t] = pred
        lw = np.log(pred + 1e-300) + B[t]; lw -= lw.max(); w = np.exp(lw); w /= w.sum()
        F[t] = w; pred = w @ A
    return F, P

def vol_order(h):
    return np.argsort(np.sqrt(h.covars_[:, 0, 0]))   # 평온 → 스트레스

def summarize(h, X, name):
    order = vol_order(h); K = h.n_components
    F, _ = forward_filter(h, X)
    rows = []
    for rank, j in enumerate(order):
        A = h.transmat_
        rows.append({"state_rank": rank, "mean_ret_%": h.means_[j, 0], "sd_ret_%": np.sqrt(h.covars_[j, 0, 0]),
                     "mean_ln_rv5": h.means_[j, 1], "typical_rv5_%": np.exp(h.means_[j, 1]),
                     "stay_prob": A[j, j], "exp_duration_days": 1 / (1 - A[j, j]), "uncond_freq": F[:, j].mean()})
    tab = pd.DataFrame(rows)
    # 전이행렬(정렬)
    A_sorted = h.transmat_[np.ix_(order, order)]
    tm = pd.DataFrame(A_sorted, index=[f"from_{i}" for i in range(K)], columns=[f"to_{j}" for j in range(K)])
    tab.to_csv(os.path.join(TAB, f"hmm_params_{name}.csv"), index=False, float_format="%.4f")
    tm.to_csv(os.path.join(TAB, f"hmm_transmat_{name}.csv"), float_format="%.4f")
    return tab, tm

out = pd.DataFrame(index=m.index)
specs = {
    "k3": (["ret", "ln_rv5"], 3),
    "k2": (["ret", "ln_rv5"], 2),
    "k4": (["ret", "ln_rv5"], 4),
    "k3dir": (["ret", "ln_rv5", "ret20"], 3),
}
models = {}
for name, (feats, k) in specs.items():
    sub = m[feats].dropna(); X = sub.values
    ll, h = fit_best(X, k)
    models[name] = (h, sub.index, feats)
    order = vol_order(h)
    F, P = forward_filter(h, X); S = h.predict_proba(X); V = h.predict(X)
    rank_of = {j: r for r, j in enumerate(order)}
    for r, j in enumerate(order):
        out.loc[sub.index, f"p_filt_{name}_s{r}"] = F[:, j]
        out.loc[sub.index, f"p_smooth_{name}_s{r}"] = S[:, j]
    out.loc[sub.index, f"viterbi_{name}"] = [rank_of[v] for v in V]
    tab, tm = summarize(h, X, name)
    print(f"\n=== {name}: feats={feats}, K={k}, logL={ll:.1f}, AIC={-2*ll + 2*(k*k - k + k*(len(feats) + len(feats)*(len(feats)+1)/2)):.1f}")
    print(tab.round(3).to_string(index=False)); print(tm.round(3).to_string())

# ---- 연도별 확장창(recursive) 재추정: 연도 Y의 filtered 확률은 Y-1년 말까지의 자료로 추정한 모수 사용 ----
feats = ["ret", "ln_rv5"]; sub = m[feats].dropna(); X = sub.values; idx = sub.index
rec = pd.Series(np.nan, index=idx, name="p_filt_k3_s2_rec")
for Y in range(2005, 2027):
    train = idx < pd.Timestamp(f"{Y}-01-01")
    res = fit_best(X[train], 3, seeds=range(3))
    if res is None: continue
    h = res[1]; order = vol_order(h)
    F, _ = forward_filter(h, X[idx < pd.Timestamp(f"{Y+1}-01-01")])
    sel = (idx >= pd.Timestamp(f"{Y}-01-01")) & (idx < pd.Timestamp(f"{Y+1}-01-01"))
    rec.loc[sel] = F[sel[idx < pd.Timestamp(f"{Y+1}-01-01")], order[-1]]
out["p_filt_k3_s2_rec"] = rec

# ---- 스트레스 지표 ----
out["lvl_pct3y"] = m["lvl_pct3y"]; out["usdkrw"] = m["usdkrw"]
out["stress_hmm"] = (out["p_filt_k3_s2"] > 0.5).astype(float)                       # 당일 filtered
for q, nm in [(0.8, "20"), (0.9, "10"), (0.7, "30")]:
    out[f"stress_{nm}"] = ((out["p_filt_k3_s2"] > 0.5) & (out["lvl_pct3y"] >= q)).astype(float)
    out[f"risk_set_{nm}"] = out[f"stress_{nm}"].shift(1)                              # 전일 정보 기준 위험집합
out["risk_set_hmm"] = out["stress_hmm"].shift(1)
out["risk_set_rec20"] = (((out["p_filt_k3_s2_rec"] > 0.5) & (out["lvl_pct3y"] >= 0.8)).astype(float)).shift(1)
out["risk_set_k4_20"] = (((out["p_filt_k4_s3"] > 0.5) & (out["lvl_pct3y"] >= 0.8)).astype(float)).shift(1)
out["risk_set_k2_20"] = (((out["p_filt_k2_s1"] > 0.5) & (out["lvl_pct3y"] >= 0.8)).astype(float)).shift(1)
out["risk_set_dir20"] = (((out["p_filt_k3dir_s2"] > 0.5) & (out["lvl_pct3y"] >= 0.8)).astype(float)).shift(1)
# 대안 정의: 비평온(변동성 확대 ∪ 스트레스) ∧ 고환율 — 2025~26년처럼 2008년 대비 변동성이 낮은 스트레스 국면을 포착
out["stress_nc"] = (out["p_filt_k3_s0"] <= 0.5).astype(float)
for q, nm in [(0.8, "20"), (0.9, "10"), (0.7, "30")]:
    out[f"risk_set_nc{nm}"] = ((out["stress_nc"] == 1) & (out["lvl_pct3y"] >= q)).astype(float).shift(1)
out["stress_k2"] = (out["p_filt_k2_s1"] > 0.5).astype(float); out["stress_dir"] = (out["p_filt_k3dir_s2"] > 0.5).astype(float)
out["stress_k4"] = (out["p_filt_k4_s3"] > 0.5).astype(float); out["stress_rec"] = (out["p_filt_k3_s2_rec"] > 0.5).astype(float)
out["stress_lvl"] = (out["lvl_pct3y"] >= 0.8).astype(float); out["risk_set_lvl20"] = out["stress_lvl"].shift(1)   # 수준만(국면 무관) — 비교용
out.index.name = "date"
out.to_csv(os.path.join(PROC, "regimes.csv"), float_format="%.5f")

# ---- 스트레스 에피소드 목록 (stress_20 기준, 3일 이하 단절은 병합) ----
s = out["stress_20"].fillna(0).values; dates = out.index
eps = []; i = 0; n = len(s)
while i < n:
    if s[i] == 1:
        j = i
        while j + 1 < n and (s[j + 1] == 1 or (s[j + 1:j + 4] == 1).any()): j += 1
        eps.append((dates[i], dates[j], j - i + 1)); i = j + 1
    else: i += 1
ep = pd.DataFrame(eps, columns=["start", "end", "days"])
ep = ep[ep["days"] >= 3].reset_index(drop=True)
ep["krw_start"] = out.loc[ep["start"], "usdkrw"].values; ep["krw_peak"] = [out.loc[a:b, "usdkrw"].max() for a, b in zip(ep["start"], ep["end"])]
ep["krw_end"] = out.loc[ep["end"], "usdkrw"].values
ep.to_csv(os.path.join(TAB, "stress_episodes_k3_top20.csv"), index=False, float_format="%.1f")
print("\n=== 고환율 스트레스 에피소드(3상태, 상위20%, 3일 이상) ===")
print(ep.to_string(index=False))
print("\n일수 요약: stress_hmm=%d, stress_20=%d, stress_10=%d, stress_30=%d, nc20=%d, lvl20=%d (총 %d일)" % (out["stress_hmm"].sum(), out["stress_20"].sum(), out["stress_10"].sum(), out["stress_30"].sum(), out["risk_set_nc20"].sum(), out["risk_set_lvl20"].sum(), len(out)))
print("recursive vs full-sample stress agreement:", round(((out["p_filt_k3_s2_rec"] > 0.5) == (out["p_filt_k3_s2"] > 0.5))[out["p_filt_k3_s2_rec"].notna()].mean(), 3))

# ---- 그림 1 ----
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
ax[0].plot(out.index, out["usdkrw"], color="#1f4e79", lw=0.8); ax[0].set_ylabel("USD/KRW")
for a, b in zip(ep["start"], ep["end"]): ax[0].axvspan(a, b, color="#d62728", alpha=0.18, lw=0)
ax[0].set_title("원/달러 환율과 고환율 스트레스 국면(3상태 HMM 스트레스 상태 ∧ 3년 분포 상위 20%)")
ax[1].fill_between(out.index, 0, out["p_filt_k3_s2"], color="#d62728", alpha=0.6, lw=0); ax[1].set_ylabel("P(stress | t)"); ax[1].set_ylim(0, 1)
ax[2].fill_between(out.index, 0, out["p_filt_k3_s1"], color="#ff7f0e", alpha=0.6, lw=0); ax[2].set_ylabel("P(elevated | t)"); ax[2].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_regimes.png"), dpi=140); plt.close()
print("saved fig1")
