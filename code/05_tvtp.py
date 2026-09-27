"""
05_tvtp.py — 비동질(시변) 전이확률 마르코프 전환모형 (TVTP-MS, Diebold–Lee–Weinbach 1994 유형)
-----------------------------------------------------------------------------------------------
y_t = [ret_t, ln rv5_t] ~ N(mu_s, Sigma_s), s_t ∈ {0 평온, 1 변동성 확대, 2 스트레스}
P(s_t=j | s_{t-1}=i, x_t) = softmax_j(a_ij + b_ij x_t),  a_ii = b_ii = 0 (기준범주 = 잔류)
x_t = 당일 구두개입(순수, 원화약세 억제) 지표. 기울기 b는 상태 1·2 행에만 허용(평온 행은 동질).
직접 최우추정(L-BFGS-B), 초기값 = 02_regimes.py의 동질 HMM. LR 검정: b=0 (동질) 대 TVTP.
위약: x_t = 원화강세 억제 발언(D_app) — 약세 스트레스 이탈과 무관해야 함.
출력: output/tables/tvtp_results.csv, output/figures/fig6_tvtp.png
"""
import os, json, warnings
LAG = int(os.environ.get("LAG", "0"))   # x_t = D_{t-LAG}: LAG=1이면 전일 발언이 익일 전이에 미치는 효과(당일 역인과 회피)
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
from hmmlearn.hmm import GaussianHMM
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib  # 한글 폰트(NanumGothic)
except ImportError:
    pass
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed"); EVD = os.path.join(ROOT, "data", "events"); TAB = os.path.join(ROOT, "output", "tables"); FIG = os.path.join(ROOT, "output", "figures")
m = pd.read_csv(os.path.join(PROC, "master_daily.csv"), parse_dates=["date"]).set_index("date").loc["2000-01-03":]
e = pd.read_csv(os.path.join(EVD, "events_daily.csv"), parse_dates=["date"]).set_index("date")
m = m.join(e[["D_dep_pure", "D_dep_hq", "D_dep", "D_app"]], how="left").fillna({"D_dep_pure": 0, "D_dep_hq": 0, "D_dep": 0, "D_app": 0})
m["ln_rv5"] = np.log(m["rv5"]); d = m[["ret", "ln_rv5", "D_dep_pure", "D_dep_hq", "D_dep", "D_app"]].dropna()
Y = d[["ret", "ln_rv5"]].values; T = len(Y); K = 3

# ---- 초기값: 동질 HMM ----
best = None
for s in range(6):
    h = GaussianHMM(n_components=K, covariance_type="full", n_iter=2000, tol=1e-6, random_state=s).fit(Y)
    ll = h.score(Y)
    if best is None or ll > best[0]: best = (ll, h)
h = best[1]; order = np.argsort(np.sqrt(h.covars_[:, 0, 0]))
mu0 = h.means_[order]; Sig0 = h.covars_[order]; A0 = h.transmat_[np.ix_(order, order)]

def pack(mu, Sig, A, b):
    L = [np.linalg.cholesky(Sig[k]) for k in range(K)]
    chol = np.concatenate([[L[k][0, 0], L[k][1, 0], L[k][1, 1]] for k in range(K)])
    a = []
    for i in range(K):
        for j in range(K):
            if j != i: a.append(np.log(A[i, j] / A[i, i]))
    return np.concatenate([mu.ravel(), np.log(np.abs(chol)) * np.array([1, 0, 1] * K) + chol * np.array([0, 1, 0] * K), a, b])

def unpack(th):
    mu = th[:6].reshape(K, 2); c = th[6:15]; a = th[15:21]; b = th[21:25]
    Sig = []
    for k in range(K):
        L = np.array([[np.exp(c[3 * k]), 0.0], [c[3 * k + 1], np.exp(c[3 * k + 2])]]); Sig.append(L @ L.T)
    return mu, np.array(Sig), a, b

def trans_mats(a, b, x):
    """x: (T,) 공변량 → (T,K,K) 전이행렬. 기울기는 행 1,2의 두 비잔류 원소에 적용"""
    logits = np.zeros((len(x), K, K)); idx = 0; bidx = 0
    for i in range(K):
        for j in range(K):
            if j != i:
                logits[:, i, j] = a[idx]
                if i >= 1: logits[:, i, j] += b[bidx] * x; bidx += 1
                idx += 1
    logits -= logits.max(axis=2, keepdims=True); P = np.exp(logits); return P / P.sum(axis=2, keepdims=True)

def loglik_frames(mu, Sig):
    B = np.zeros((T, K))
    for k in range(K):
        Si = np.linalg.inv(Sig[k]); det = np.linalg.det(Sig[k]); diff = Y - mu[k]
        B[:, k] = -0.5 * (np.einsum("ij,jk,ik->i", diff, Si, diff) + np.log(det) + 2 * np.log(2 * np.pi))
    return B

def negll(th, x, use_b=True):
    mu, Sig, a, b = unpack(th)
    if not use_b: b = np.zeros(4)
    Pt = trans_mats(a, b, x); B = loglik_frames(mu, Sig)
    Bm = B.max(axis=1, keepdims=True); E = np.exp(B - Bm)
    pred = np.full(K, 1.0 / K); ll = 0.0
    for t in range(T):
        w = pred * E[t]; c = w.sum(); ll += np.log(c) + Bm[t, 0]; w /= c; pred = w @ Pt[t]
    return -ll

def fit(x, use_b=True, th0=None):
    th0 = pack(mu0, Sig0, A0, np.zeros(4)) if th0 is None else th0
    res = minimize(negll, th0, args=(x, use_b), method="L-BFGS-B", options={"maxiter": 400, "maxfun": 40000})
    return res

results = {}; rows = []
x0 = np.zeros(T)
r0 = fit(x0, use_b=False); ll0 = -r0.fun
for name in ["D_dep_pure", "D_dep_hq", "D_dep", "D_app"]:
    x = d[name].shift(LAG).fillna(0).values.astype(float)
    r1 = fit(x, use_b=True, th0=r0.x); ll1 = -r1.fun
    mu, Sig, a, b = unpack(r1.x)
    P0 = trans_mats(a, b, np.array([0.0]))[0]; P1 = trans_mats(a, b, np.array([1.0]))[0]
    lr = 2 * (ll1 - ll0); pv = 1 - chi2.cdf(lr, 4)
    results[name] = {"ll_tvtp": ll1, "ll_homog": ll0, "LR": lr, "p_LR": pv, "n_x1": int(x.sum()),
                     "P_stress_stay_x0": P0[2, 2], "P_stress_stay_x1": P1[2, 2], "P_stress_to_nonstress_x0": 1 - P0[2, 2], "P_stress_to_nonstress_x1": 1 - P1[2, 2],
                     "P_elev_to_stress_x0": P0[1, 2], "P_elev_to_stress_x1": P1[1, 2], "P_elev_to_calm_x0": P0[1, 0], "P_elev_to_calm_x1": P1[1, 0], "b": b.tolist()}
    print(f"{name}: n(x=1)={int(x.sum())} LR={lr:.2f} p={pv:.3f} | stress stay: {P0[2,2]:.3f} -> {P1[2,2]:.3f} | elev->stress: {P0[1,2]:.3f} -> {P1[1,2]:.3f} | elev->calm: {P0[1,0]:.3f} -> {P1[1,0]:.3f}")
    rows.append({"covariate": name, **{k: v for k, v in results[name].items() if k != "b"}})
tab = pd.DataFrame(rows); tab["lag"] = LAG; tab.to_csv(os.path.join(TAB, f"tvtp_results_lag{LAG}.csv"), index=False, float_format="%.4f")
with open(os.path.join(ROOT, "output", f"tvtp_results_lag{LAG}.json"), "w") as f: json.dump(results, f, indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
names = ["D_dep_pure", "D_dep_hq", "D_dep", "D_app"]; xx = np.arange(len(names))
ax[0].bar(xx - 0.2, [results[n]["P_stress_to_nonstress_x0"] for n in names], 0.4, label="발언 없음", color="#9ecae1")
ax[0].bar(xx + 0.2, [results[n]["P_stress_to_nonstress_x1"] for n in names], 0.4, label="발언일", color="#d62728")
ax[0].set_xticks(xx); ax[0].set_xticklabels(names, rotation=15); ax[0].set_title(f"P(스트레스 → 비스트레스), 1일 전이 (발언 시차 {LAG}일)"); ax[0].legend()
ax[1].bar(xx - 0.2, [results[n]["P_elev_to_stress_x0"] for n in names], 0.4, label="발언 없음", color="#9ecae1")
ax[1].bar(xx + 0.2, [results[n]["P_elev_to_stress_x1"] for n in names], 0.4, label="발언일", color="#d62728")
ax[1].set_xticks(xx); ax[1].set_xticklabels(names, rotation=15); ax[1].set_title("P(변동성 확대 → 스트레스), 1일 전이"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, f"fig6_tvtp_lag{LAG}.png"), dpi=140); plt.close(); print("saved fig6")
