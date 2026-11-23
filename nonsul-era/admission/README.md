# 대학별 수시 모집요강 논술전형 DB (2025·2026·2027 + 2028 시행계획)

대상: 기출 DB 44개교 + 2028 논술 신설 3곳(대전대·한성대·한양대 ERICA, 자료 있는 연도만).
원본: 대교협·서울시교육청 공시 CDN(cdn013.negagea.net, univ_info{Y-1}/학교/학교_{Y}학년도_수시모집요강.pdf),
2028은 `학교_2028학년도_대학입학전형계획.pdf`, 2025는 각 대학 입학처 공식 PDF. 학원·언론 자료 사용 안 함.

## 파일
- `schools/<CODE>.yaml` : 학교 1곳 = 파일 1개 (CODE는 기출 DB와 같은 학교 코드)
- `sources.tsv` : 학교·연도별 원본 URL, 쪽수
- `build_csv.py` → `summary.csv` (학교×연도 1줄 요약), `changes.csv` (연도 간 변화 자동 비교)

## 스키마 (years[] 한 항목 = 한 학년도 문서)
```yaml
school: 경희대학교
code: KHU
years:
  - year: 2027                 # 학년도
    doc: 수시모집요강            # 또는 대학입학전형계획(2028)
    url: https://...
    pages: "45-52"              # 논술 부분 PDF 쪽
    nonsul: true                # 그 해 논술전형 실시 여부
    jeonhyeong: 논술우수전형      # 전형명(원문)
    total_quota: 491            # 논술전형 총 모집인원
    by_track: {인문: 0, 자연: 0, 의약: 0}   # 원문 계열 구분 그대로 키 사용
    units:                      # 모집단위별 인원(논술 응시 계열 표시)
      - {name: 수학과, college: 이과대학, track: 자연, quota: 7}
    ratio: {논술: 100}          # 전형요소 반영비율 (학생부교과 등 원문 이름)
    student_record: "학생부 반영 방식 요약(없으면 null)"
    csat_min:                   # 수능 최저 원문 거의 그대로
      - {applies: 자연계열, rule: "국,수,영,탐(1) 중 2개 합 5 이내", note: "..."}
    schedule:                   # 논술고사 일정
      - {date: 2026-11-21, time: "09:00~10:40", applies: "자연계열(의학 제외)"}
    exam:                       # 계열/유형별 시험 구성
      - {type: 자연계, minutes: 100, items: "3문항(소문항 포함)", answer_form: 서술형, subjects: [수학Ⅰ, 수학Ⅱ, 미적분], science: null, note: ""}
    changes: ["요강에 적힌 전년 대비 변경사항 원문 요약"]
    notes: ""
```
선택 필드: units의 `type`(같은 모집단위 안 논술 유형 구분), `campus`, `note`; 연도 단위 `by_type`, `by_campus`, `missing`(원본 미확보).
값이 문서에 없으면 `null`, 판독 애매하면 note에 `[판독확인]`.
