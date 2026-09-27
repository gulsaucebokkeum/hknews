# 고환율 스트레스 국면에서 구두개입의 국면전환 효과 — 파일럿 저장소

논문 공모전용 연구안의 **설계 검토·데이터 탐색·모델링·초안**을 담은 저장소입니다.

| 경로 | 내용 |
|---|---|
| `docs/design_review_ko.md` | 연구설계 검토 의견서(피드백) |
| `docs/codebook_events_ko.md` | 구두개입 사건 코딩북 |
| `paper/draft_ko.md` | 논문 초안(파일럿 결과 포함) |
| `data/raw/` | 원자료(출처는 `data/raw/SOURCES.md`) |
| `data/events/raw/` | 기간별 사건 수집 파일과 수집 노트 |
| `data/events/events_master.csv`, `events_daily.csv` | 정제된 사건표·일별 처치 패널 |
| `data/processed/master_daily.csv` | 일별 마스터 데이터(1999–2026) |
| `data/processed/regimes.csv` | HMM 국면 확률·스트레스 지표 |
| `code/01_build_data.py` → `05_tvtp.py` | 재현 파이프라인 |
| `output/tables/`, `output/figures/`, `output/logs/` | 결과표·그림·실행 로그 |

## 재현
```bash
pip install -r requirements.txt
python3 code/01_build_data.py      # 마스터 데이터
python3 code/02_regimes.py         # HMM 국면(3상태 기본 + 강건성)
python3 code/03_events.py          # 사건 병합·처치변수
RISK_SET=risk_set_20 python3 code/04_analysis.py    # 계획서 정의(최상위 변동성 상태 ∧ 3년 상위 20%)
RISK_SET=risk_set_nc20 python3 code/04_analysis.py  # 대안 정의(비평온 ∧ 상위 20%)
python3 code/05_tvtp.py            # 비동질 전이확률 MS 모형
```
환경변수: `RISK_SET`(위험집합 정의), `B_BOOT`(블록 부트스트랩 횟수), `N_PLACEBO`(위약 반복), `START`(표본 시작일).

## 주의
- 환율 주 계열은 계획서의 ECOS 서울 종가가 아니라 FRED DEXKOUS(뉴욕 정오)이며 ECB 교차환율로 보완했습니다(네트워크 제약). 최종본에서는 ECOS로 교체를 권고합니다.
- 사건 자료는 공개 검색 요약으로 구축된 **파일럿**이며 신뢰도(high/medium/low) 플래그가 있습니다. `low` 행은 원문 대조 전 인용 불가입니다.
