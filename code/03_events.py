"""
03_events.py — 구두개입 사건 DB 병합·정제 및 일별 처치변수 생성
------------------------------------------------------------------
입력: data/events/raw/events_*.csv (기간별 수집 파일; 코드북은 docs/codebook_events_ko.md)
출력: data/events/events_master.csv  (사건 단위, 거래일 매핑·군집 정보 포함)
      data/events/events_daily.csv   (거래일 패널: D_dep, D_dep_pure, intensity 등)
      output/tables/events_summary_year.csv
"""
import glob, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "events", "raw"); OUT = os.path.join(ROOT, "data", "events"); TAB = os.path.join(ROOT, "output", "tables")
os.makedirs(TAB, exist_ok=True)
COLS = ["date", "time_kst", "speaker", "title", "institution", "statement", "direction", "intensity", "joint", "official",
        "actual_intervention_reported", "usdkrw_level", "source_urls", "confidence", "notes"]

frames = []
for f in sorted(glob.glob(os.path.join(RAW, "events_*.csv"))):
    d = pd.read_csv(f, dtype=str, encoding="utf-8-sig", on_bad_lines="warn", engine="python")
    d.columns = [c.strip().lower() for c in d.columns]
    for c in COLS:
        if c not in d.columns: d[c] = np.nan
    d = d[COLS].copy(); d["src_file"] = os.path.basename(f); frames.append(d)
ev = pd.concat(frames, ignore_index=True)
ev["date"] = pd.to_datetime(ev["date"].str.strip(), errors="coerce")
ev = ev[ev["date"].notna()].copy()
ev = ev[(ev["date"] >= "2000-01-01") & (ev["date"] <= "2026-09-30")]
for c in ["direction", "joint", "official", "actual_intervention_reported", "confidence"]:
    ev[c] = ev[c].fillna("").str.strip().str.lower()
ev["direction"] = ev["direction"].replace({"curb_dep": "curb_depreciation", "curb_app": "curb_appreciation", "": "unclear"})
ev["intensity"] = pd.to_numeric(ev["intensity"], errors="coerce").fillna(1).clip(1, 3).astype(int)
ev["joint"] = ev["joint"].str[:1].eq("y"); ev["official"] = ev["official"].str[:1].eq("y")
ev["actual"] = ev["actual_intervention_reported"].str[:1].map({"y": "Y", "n": "N"}).fillna("U")
ev["conf_rank"] = ev["confidence"].map({"high": 3, "medium": 2, "low": 1}).fillna(1)
ev["placeholder"] = ev["notes"].fillna("").str.contains("PLACEHOLDER|placeholder|month-only|1st of the month|month only|월 단위", case=False, regex=True) | \
                    ev["statement"].fillna("").str.contains("PLACEHOLDER", case=False)
ev["usdkrw_level"] = pd.to_numeric(ev["usdkrw_level"].astype(str).str.replace(",", "").str.extract(r"(\d{3,4}\.?\d*)")[0], errors="coerce")

# ---- 같은 날·같은 방향 사건 통합 ----
def agg(g):
    return pd.Series({
        "n_statements": len(g), "intensity": g["intensity"].max(), "joint": g["joint"].any(), "official": g["official"].any(),
        "actual": "Y" if (g["actual"] == "Y").any() else ("N" if (g["actual"] == "N").any() else "U"),
        "speakers": " | ".join(sorted(set(g["speaker"].dropna().astype(str)))),
        "institutions": " | ".join(sorted(set(g["institution"].dropna().astype(str)))),
        "statement": " || ".join(g["statement"].dropna().astype(str).str[:300]),
        "time_kst": ";".join(g["time_kst"].dropna().astype(str)),
        "usdkrw_level": g["usdkrw_level"].max(), "confidence": g["conf_rank"].max(), "placeholder": g["placeholder"].all(),
        "source_urls": " ; ".join(g["source_urls"].dropna().astype(str)), "src_files": ",".join(sorted(set(g["src_file"]))),
    })
evd = ev.groupby(["date", "direction"]).apply(agg).reset_index()

# ---- 거래일 캘린더 매핑 (비거래일 사건은 다음 거래일로) ----
cal = pd.read_csv(os.path.join(ROOT, "data", "processed", "master_daily.csv"), usecols=["date"], parse_dates=["date"])["date"]
cal = cal[cal >= "2000-01-01"].reset_index(drop=True)
pos = np.searchsorted(cal.values, evd["date"].values, side="left")
pos = np.minimum(pos, len(cal) - 1)
evd["trade_date"] = cal.values[pos]
evd["date_shifted"] = (evd["trade_date"] != evd["date"])
evd["tidx"] = pos

# ---- 군집(반복 개입) 정의: 같은 방향, 직전 사건과 간격 ≤ g 거래일 ----
evd = evd.sort_values(["direction", "trade_date"]).reset_index(drop=True)
for g in (3, 5, 10):
    cid = np.zeros(len(evd), dtype=int); seq = np.zeros(len(evd), dtype=int); c = 0
    for d_ in evd["direction"].unique():
        idx = evd.index[evd["direction"] == d_]; prev = None; k = 0
        for i in idx:
            if prev is None or evd.at[i, "tidx"] - prev > g:
                c += 1; k = 1
            else:
                k += 1
            cid[i] = c; seq[i] = k; prev = evd.at[i, "tidx"]
    evd[f"cluster_{g}"] = cid; evd[f"seq_{g}"] = seq
evd["repeated_5"] = evd["seq_5"] > 1
evd = evd.sort_values("trade_date").reset_index(drop=True)
evd.to_csv(os.path.join(OUT, "events_master.csv"), index=False)

# ---- 일별 패널 ----
daily = pd.DataFrame(index=cal.values); daily.index.name = "date"
dep = evd[evd["direction"].isin(["curb_depreciation", "two_sided"])].drop_duplicates("trade_date").set_index("trade_date")
app = evd[evd["direction"] == "curb_appreciation"].drop_duplicates("trade_date").set_index("trade_date")
daily["D_dep"] = 0; daily.loc[dep.index[~dep["placeholder"]], "D_dep"] = 1
daily["D_dep_pure"] = 0; daily.loc[dep.index[(dep["actual"] != "Y") & (~dep["placeholder"])], "D_dep_pure"] = 1
daily["D_dep_hq"] = 0; daily.loc[dep.index[(dep["actual"] != "Y") & (~dep["placeholder"]) & (dep["confidence"] >= 2)], "D_dep_hq"] = 1
daily["D_dep_actual"] = 0; daily.loc[dep.index[dep["actual"] == "Y"], "D_dep_actual"] = 1
daily["D_dep_official"] = 0; daily.loc[dep.index[dep["official"]], "D_dep_official"] = 1
daily["D_dep_joint"] = 0; daily.loc[dep.index[dep["joint"]], "D_dep_joint"] = 1
daily["D_dep_hi"] = 0; daily.loc[dep.index[dep["intensity"] >= 2], "D_dep_hi"] = 1
daily["D_dep_first5"] = 0; daily.loc[dep.index[dep["seq_5"] == 1], "D_dep_first5"] = 1
daily["D_dep_rep5"] = 0; daily.loc[dep.index[dep["seq_5"] > 1], "D_dep_rep5"] = 1
daily["intensity_dep"] = 0; daily.loc[dep.index, "intensity_dep"] = dep["intensity"].values
daily["conf_dep"] = 0; daily.loc[dep.index, "conf_dep"] = dep["confidence"].values
daily["level_at_stmt"] = np.nan; daily.loc[dep.index, "level_at_stmt"] = dep["usdkrw_level"].values
daily["D_app"] = 0; daily.loc[app.index, "D_app"] = 1
daily["actual_any"] = 0; daily.loc[evd.index[evd["actual"] == "Y"].map(lambda i: evd.at[i, "trade_date"]), "actual_any"] = 1
daily["n_dep_past20"] = daily["D_dep"].rolling(20).sum().shift(1).fillna(0)
daily.to_csv(os.path.join(OUT, "events_daily.csv"))

# ---- 요약표 ----
evd["year"] = evd["trade_date"].dt.year
summ = evd.pivot_table(index="year", columns="direction", values="trade_date", aggfunc="count", fill_value=0)
summ["dep_pure"] = evd[(evd["direction"] == "curb_depreciation") & (evd["actual"] != "Y") & (~evd["placeholder"])].groupby("year").size()
summ["dep_hq"] = evd[(evd["direction"] == "curb_depreciation") & (evd["actual"] != "Y") & (~evd["placeholder"]) & (evd["confidence"] >= 2)].groupby("year").size()
summ["dep_official"] = evd[(evd["direction"] == "curb_depreciation") & (evd["official"])].groupby("year").size()
summ = summ.fillna(0).astype(int); summ.loc["total"] = summ.sum()
summ.to_csv(os.path.join(TAB, "events_summary_year.csv"))
print(summ.to_string())
print("\n원본 발언 수:", len(ev), " 일·방향 통합 사건 수:", len(evd), " 비거래일→이동:", int(evd["date_shifted"].sum()))
print("방향별:", evd["direction"].value_counts().to_dict())
print("약세억제 사건 중 실개입 보도 Y/N/U:", dep["actual"].value_counts().to_dict())
print("placeholder 날짜 사건:", int(evd["placeholder"].sum()), " 신뢰도 분포(사건):", evd["confidence"].value_counts().sort_index().to_dict())
print("반복(5일 규칙) 비율:", round(dep["seq_5"].gt(1).mean(), 3), " 강도 분포:", dep["intensity"].value_counts().sort_index().to_dict())
