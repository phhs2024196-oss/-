---
# ── 식별 ──────────────────────────────────────────────
id: SSU-T01-011            # 학교-유형-순번(3자리). 한 번 부여하면 변경·재사용 금지
legacy_id: null            # 기존 300문항이면 S01-01 형식
kind: original             # past | mock | original | variant
school: SSU
type: T01
parent: null               # variant 일 때 원 문항 ID (예: SSU-G26-04)

# ── 분류 ──────────────────────────────────────────────
level: L5                  # L3~L6 (막히는 지점 수 기준)
form: 분류                 # taxonomy.yaml forms 12종 중 1개
points: 25                 # 문항 총점
sub_points: []             # 다소문항이면 [8, 10, 12] — 합이 points 와 같아야 함
subjects: [수학Ⅱ]          # 2015 개정 과목명 (풀이에 필요한 모든 과목)
scope_2028: common         # common | calc2 | geom | unknown
scope_confirmed: true      # false = 자동 추정값(이관 문항 기본). 사람/검토 모델이 확인하면 true
concepts: [다항함수의 차수, 항등식]
key_idea: "한 줄 열쇠"
stuck: "학생이 막히는 지점(AI 첨삭 힌트 근거)"
answer: "$19$"

# ── 출처·저작권 (past/mock/variant 필수) ───────────────
source: null               # {school: SSU, year: 2026, session: null, number: 4, url: "..."}
copyright:
  original_text: false     # 대학 원문 그대로 수록 허락을 받았는가

# ── 검수 ──────────────────────────────────────────────
status: draft              # draft → r1_pass → r2_pass → verified → (sampled) → published
verify: verify/SSU-T01-s2.py::p11   # 세트 2(011~020) 검산 파일의 p11
reviews: []                # - {stage: r1, reviewer: claude-sonnet-5-5, date: 2026-12-01, result: pass, note: ""}
version: 1
created: 2026-12-01
updated: 2026-12-01
---

## 제시문

(세종대는 이 절을 비워 둔다. 숭실: 교과서 정리 1~2개 + 【출처: 과목 「단원」】. 동국: 【가】~【다】)

## 논제

(동국·세종: (1)(2)(3) 소문항과 소문항별 배점을 [8점] / (80점) 형식으로 명시)

## 모범답안

① ...

② ...

## 정답

$19$

## 채점표

| 단계 | 소문항 | 내용 | 배점 |
|---|---|---|---|
| ① | - | ... | 6 |
| ② | - | ... | 7 |
| ③ | - | ... | 12 |

(단일 논제는 소문항 칸에 `-`. 다소문항은 (1)(2)(3)을 쓰고 ①②③…은 끝까지 이어서 번호. 배점은 정수, 합 = points)

## 대표 오답

어떤 실수 → 어떤 값.

## 메모

(선택: 출제 의도, 교재 배치 메모, 검토 중 바뀐 점)
