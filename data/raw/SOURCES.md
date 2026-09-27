# 원자료 출처 및 취득 경로 (2026-09-27 기준)

| 파일 | 내용 | 원출처 | 취득 경로 | 비고 |
|---|---|---|---|---|
| fred_h10_daily.csv | 미 연준 H.10 일별 환율(22개국, long 포맷). 원/달러=DEXKOUS(뉴욕 정오 매입률), 위안/달러=DEXCHUS, 유로·엔·파운드·캐나다달러·스웨덴크로나·스위스프랑 | Federal Reserve H.10 / FRED | 공개 GitHub 저장소 `datasets/exchange-rates` (Frictionless Data, PDDL) | 데이터셋은 모든 통화를 '외화/달러' 단위로 저장(유로·파운드는 공식 H.10과 반대 방향) → 코드에서 반전 |
| ecb_eurofxref_hist.csv | ECB 유로 기준환율(14:15 CET 고시) 1999–2026-09-14. EUR/KRW, EUR/USD, EUR/CNY(2005-04~) 등 | European Central Bank | PyPI 패키지 `CurrencyConverter` 0.18.22에 동봉된 `eurofxref-hist.zip` | 원/달러 교차환율 = (EUR/KRW)/(EUR/USD). 미국 휴일의 결측 보충 및 교차검증용 |
| cboe_vix_daily.csv | CBOE VIX 일별 OHLC 1990–2026-09-22 | CBOE | 공개 GitHub `datasets/finance-vix` | |
| eia_wti_daily.csv, eia_brent_daily.csv | WTI·Brent 현물 일별 1986/1987–2026-09-22 | US EIA | 공개 GitHub `datasets/oil-prices` | 2020-04-20 WTI 음수 |
| fred_dgs2_1976_2026.csv, fred_dgs10_1976_2026.csv | 미 국채 2년·10년 만기수익률(DGS2, DGS10) 1976–2026-07-30 | FRED / Fed H.15 | 공개 GitHub `TGRADEA/gradea-fred-archive` (FRED 미러) | 2026-08~09 결측 |
| kospi_krx_2008_2026.csv | KOSPI 종가·거래량·시가총액 2008-01-02–2026-09-23 | KRX (pykrx 수집) | 공개 GitHub `prefire-resource/kospi-index` | 2026년 값(급등락)은 별도 검증 필요 |
| kospi_yahoo_1997_2017.csv | KOSPI OHLC 1997-07–2017-07 | Yahoo Finance | 공개 GitHub `gomjellie/kospi-kosdaq-csv` | 2008년 이전 구간 보충용 |
| krx_investor_flows_2008_2026.csv | KOSPI 투자자별 순매수(원) 2008–2026 | KRX (pykrx 수집) | 공개 GitHub `prefire-resource/kospi-index` | 외국인 순매수(억원)로 변환 |

## 계획서 대비 차이 및 한계
- 계획서는 ECOS 은행간 **서울 종가**를 주 계열로 지정했으나, 본 세션 환경에서는 ECOS·bok.or.kr·FRED·Yahoo 등 대부분의 외부 호스트가 네트워크 정책으로 차단되어 접근 불가. 대신 공개 미러를 통해 **FRED DEXKOUS(뉴욕 정오)** 를 주 계열, **ECB 교차환율(14:15 CET)** 을 보조 계열로 사용.
- 두 fix는 모두 서울 장 마감(15:30 KST) 이후 시점이므로, 당일(t) 서울 장중의 구두개입은 두 fix 모두에 **같은 날짜(t)** 에 반영된다. 즉 t−1 fix는 처치 이전, t fix는 처치 이후 값이다. (ECB fix는 22:15 KST, 뉴욕 정오는 익일 01~02시 KST. 예: 2024-12-03 계엄(22:27 KST)은 ECB fix에는 12-03에 미반영, FRED 12-03에는 반영.)
- 최종 논문에서는 ECOS 서울 종가(계열 731Y001)로 교체하고, 본 초안 결과와의 일치 여부를 보고할 것을 권고.
- DXY는 ICE 공식(EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%)으로 H.10 환율에서 합성한 것이며 ICE 선물 종가와는 시점 차이가 있다.
- `prefire-resource` 저장소의 `fx_usdkrw.csv`는 ECB 교차환율과 동일한 것으로 확인되어 사용하지 않았다.
