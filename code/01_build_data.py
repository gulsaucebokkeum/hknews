"""
01_build_data.py — 일별 마스터 데이터셋 구축
------------------------------------------------
입력(data/raw):
  fred_h10_daily.csv        : FRED/Fed H.10 일별 환율(국가별 long 포맷). 원/달러=DEXKOUS(뉴욕 정오 매입률)
  ecb_eurofxref_hist.csv    : ECB 유로 기준환율(14:15 CET). KRW/USD 교차환율로 원/달러 보조계열
  cboe_vix_daily.csv        : CBOE VIX (OHLC)
  eia_wti_daily.csv, eia_brent_daily.csv : EIA 유가
  fred_dgs2*.csv, fred_dgs10*.csv : 미 국채 2년/10년 (가용 구간)
  kospi_*.csv               : KOSPI 종가 (가용 구간)
출력(data/processed/master_daily.csv)
"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

START, END = "1999-01-04", "2026-09-30"

# ---------- 1. FRED H.10 ----------
h10 = pd.read_csv(os.path.join(RAW, "fred_h10_daily.csv"))
h10["Date"] = pd.to_datetime(h10["Date"])
wide = h10.pivot(index="Date", columns="Country", values="Exchange rate").sort_index()
wide = wide.loc[START:END]
# H.10 단위: 대부분 '외화/달러'. Euro, United Kingdom, Australia, New Zealand는 '달러/외화'.
fred = pd.DataFrame(index=wide.index)
fred["usdkrw_fred"] = wide["South Korea"]
fred["usdcny"] = wide["China"]
fred["usdjpy"] = wide["Japan"]
fred["eurusd"] = 1.0 / wide["Euro"]          # 데이터셋은 EUR per USD로 저장 → 반전
fred["gbpusd"] = 1.0 / wide["United Kingdom"]  # GBP per USD → 반전
fred["usdcad"] = wide["Canada"]
fred["usdsek"] = wide["Sweden"]
fred["usdchf"] = wide["Switzerland"]
# 아시아 통화 바스켓(달러 대비, 등가중 로그변화 누적): TWD, SGD, THB, MYR, INR, JPY — 원화의 "상대적 이상반응" 계산용
asia = ["Taiwan", "Singapore", "Thailand", "Malaysia", "India", "Japan"]
asia_lr = np.log(wide[asia]).diff().mean(axis=1)
fred["asia_basket"] = np.exp(asia_lr.fillna(0).cumsum()) * 100
# ICE 달러인덱스(DXY) 공식으로 합성
fred["dxy_synth"] = (50.14348112 * fred["eurusd"] ** (-0.576) * fred["usdjpy"] ** 0.136
                     * fred["gbpusd"] ** (-0.119) * fred["usdcad"] ** 0.091
                     * fred["usdsek"] ** 0.042 * fred["usdchf"] ** 0.036)

# ---------- 2. ECB reference rates ----------
ecb = pd.read_csv(os.path.join(RAW, "ecb_eurofxref_hist.csv"))
ecb = ecb.loc[:, ~ecb.columns.str.contains("^Unnamed")]
ecb["Date"] = pd.to_datetime(ecb["Date"])
ecb = ecb.set_index("Date").sort_index().loc[START:END]
ecb_x = pd.DataFrame(index=ecb.index)
ecb_x["usdkrw_ecb"] = ecb["KRW"] / ecb["USD"]
ecb_x["usdcny_ecb"] = ecb["CNY"] / ecb["USD"]
ecb_x["dxy_synth_ecb"] = (50.14348112 * ecb["USD"] ** (-0.576) * (ecb["JPY"] / ecb["USD"]) ** 0.136
                          * (ecb["USD"] / ecb["GBP"]) ** (-0.119) * (ecb["CAD"] / ecb["USD"]) ** 0.091
                          * (ecb["SEK"] / ecb["USD"]) ** 0.042 * (ecb["CHF"] / ecb["USD"]) ** 0.036)

# ---------- 3. VIX, oil, rates, KOSPI ----------
vix = pd.read_csv(os.path.join(RAW, "cboe_vix_daily.csv"))
vix["DATE"] = pd.to_datetime(vix["DATE"])
vix = vix.set_index("DATE")[["CLOSE"]].rename(columns={"CLOSE": "vix"})

wti = pd.read_csv(os.path.join(RAW, "eia_wti_daily.csv")); wti["Date"] = pd.to_datetime(wti["Date"])
wti = wti.set_index("Date").rename(columns={"Price": "wti"})
brent = pd.read_csv(os.path.join(RAW, "eia_brent_daily.csv")); brent["Date"] = pd.to_datetime(brent["Date"])
brent = brent.set_index("Date").rename(columns={"Price": "brent"})

def load_fred_simple(pattern, name):
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, pattern))):
        d = pd.read_csv(f)
        dcol = [c for c in d.columns if c.lower() in ("observation_date", "date")][0]
        vcol = [c for c in d.columns if c != dcol][0]
        d[dcol] = pd.to_datetime(d[dcol])
        s = pd.to_numeric(d[vcol], errors="coerce")
        frames.append(pd.Series(s.values, index=d[dcol], name=name))
    if not frames:
        return pd.Series(dtype=float, name=name)
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated(keep="last")]

dgs2 = load_fred_simple("fred_dgs2*.csv", "dgs2")
dgs10 = load_fred_simple("fred_dgs10*.csv", "dgs10")

def load_close(path, name):
    d = pd.read_csv(path)
    dcol = [c for c in d.columns if c.lower().lstrip("\ufeff") == "date"][0]
    ccol = "Close" if "Close" in d.columns else [c for c in d.columns if "close" in c.lower()][0]
    d[dcol] = pd.to_datetime(d[dcol])
    s_ = pd.Series(pd.to_numeric(d[ccol], errors="coerce").values, index=d[dcol], name=name).sort_index()
    return s_[~s_.index.duplicated(keep="last")]

kospi_krx = load_close(os.path.join(RAW, "kospi_krx_2008_2026.csv"), "kospi")
kospi_yah = load_close(os.path.join(RAW, "kospi_yahoo_1997_2017.csv"), "kospi")
kospi = kospi_krx.combine_first(kospi_yah).rename("kospi")          # KRX 우선

# (prefire-resource fx_usdkrw.csv는 ECB 교차환율과 동일한 것으로 확인되어 사용하지 않음)

inv = pd.read_csv(os.path.join(RAW, "krx_investor_flows_2008_2026.csv")); inv["date"] = pd.to_datetime(inv["date"])
for_flow = pd.Series(pd.to_numeric(inv["foreign"], errors="coerce").values / 1e8, index=inv["date"], name="for_flow").sort_index()  # 억원
for_flow = for_flow[~for_flow.index.duplicated(keep="last")]

# ---------- 4. 캘린더: 원/달러 fix가 존재하는 날 (FRED 우선, 미국 휴일은 ECB로 보충) ----------
df = fred.join(ecb_x, how="outer")
df = df.loc[START:END]
df["usdkrw"] = df["usdkrw_fred"]
df["krw_src"] = np.where(df["usdkrw_fred"].notna(), "FRED",
                         np.where(df["usdkrw_ecb"].notna(), "ECB", "NA"))
df.loc[df["usdkrw"].isna(), "usdkrw"] = df.loc[df["usdkrw"].isna(), "usdkrw_ecb"]
df = df[df["usdkrw"].notna()].copy()
df["usdcny"] = df["usdcny"].fillna(df["usdcny_ecb"])
df["dxy_synth"] = df["dxy_synth"].fillna(df["dxy_synth_ecb"])
df = df.join(vix, how="left").join(wti, how="left").join(brent, how="left")
df = df.join(dgs2, how="left").join(dgs10, how="left").join(kospi, how="left").join(for_flow, how="left")
# 미국 휴일 등 결측은 직전값으로 보충(최대 3일)
for c in ["vix", "wti", "brent", "dgs2", "dgs10", "kospi", "usdcny", "dxy_synth"]:
    df[c] = df[c].ffill(limit=3)

# ---------- 5. 파생변수 ----------
df["ln_krw"] = np.log(df["usdkrw"])
df["ret"] = 100 * df["ln_krw"].diff()                      # +: 원화 약세
df["ret_ecb"] = 100 * np.log(df["usdkrw_ecb"]).diff()
df["rv5"] = np.sqrt((df["ret"] ** 2).rolling(5).mean() * 252)     # 연율화 실현변동성(%)
df["rv20"] = np.sqrt((df["ret"] ** 2).rolling(20).mean() * 252)
df["ret5"] = 100 * (df["ln_krw"] - df["ln_krw"].shift(5))
df["ret20"] = 100 * (df["ln_krw"] - df["ln_krw"].shift(20))
df["asia_basket"] = df["asia_basket"].ffill(limit=3)
df["d_asia"] = 100 * np.log(df["asia_basket"]).diff()
df["d_asia5"] = 100 * (np.log(df["asia_basket"]) - np.log(df["asia_basket"]).shift(5))
df["d_dxy"] = 100 * np.log(df["dxy_synth"]).diff()
df["d_dxy5"] = 100 * (np.log(df["dxy_synth"]) - np.log(df["dxy_synth"]).shift(5))
df["d_cny"] = 100 * np.log(df["usdcny"]).diff()
df["d_cny5"] = 100 * (np.log(df["usdcny"]) - np.log(df["usdcny"]).shift(5))
df["d_vix"] = df["vix"].diff()
df["d_vix5"] = df["vix"].diff(5)
df["ln_vix"] = np.log(df["vix"])
df["d_wti"] = 100 * np.log(df["wti"].where(df["wti"] > 0)).diff()   # 2020-04-20 음수 유가 제외
df["d_dgs2"] = df["dgs2"].diff()
df["d_dgs2_5"] = df["dgs2"].diff(5)
df["d_dgs10"] = df["dgs10"].diff()
df["d_kospi"] = 100 * np.log(df["kospi"]).diff()
df["d_kospi5"] = 100 * (np.log(df["kospi"]) - np.log(df["kospi"]).shift(5))
df["for_flow5"] = df["for_flow"].rolling(5).sum()

# 과거 3년(756거래일) 분포 대비 환율 수준 백분위 (t-1까지 정보만 사용)
lvl = df["usdkrw"].values
n = len(lvl); W = 756; MINP = 500
pct = np.full(n, np.nan)
for t in range(n):
    lo = max(0, t - W); win = lvl[lo:t]          # t-1까지
    if len(win) >= MINP:
        pct[t] = (win < lvl[t]).mean()
df["lvl_pct3y"] = pct
df["dow"] = df.index.dayofweek
df["year"] = df.index.year
df["month"] = df.index.month

df.index.name = "date"
cols = ["usdkrw", "krw_src", "usdkrw_fred", "usdkrw_ecb", "ln_krw", "ret", "ret_ecb", "rv5", "rv20", "ret5", "ret20",
        "dxy_synth", "d_dxy", "d_dxy5", "asia_basket", "d_asia", "d_asia5", "usdcny", "d_cny", "d_cny5", "vix", "ln_vix", "d_vix", "d_vix5",
        "wti", "brent", "d_wti", "dgs2", "d_dgs2", "d_dgs2_5", "dgs10", "d_dgs10", "kospi", "d_kospi", "d_kospi5", "for_flow", "for_flow5",
        "lvl_pct3y", "dow", "year", "month"]
df[cols].to_csv(os.path.join(OUT, "master_daily.csv"), float_format="%.6f")
print("rows", len(df), df.index.min().date(), df.index.max().date())
print(df[cols].describe().T[["count", "mean", "std", "min", "max"]].round(3).to_string())
print("krw_src counts:\n", df["krw_src"].value_counts())
