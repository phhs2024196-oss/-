# 기출 DB 입력 공통 지시서 (모든 학교 공통 — 학교별 지시는 5줄만 따로)

너는 '논술의 시대' 문항 DB에 기출·모의 수리논술 문항을 입력한다. 한국어. git 작업 금지(파일만 쓴다).
학교별 지시(별도 5줄): 원본 PDF 경로·쪽 범위 / ID 범위 / 학교 conf(db/schools/<학교>.yaml) / 오개념 ID 범위 / 특이사항.

## 순서 (토큰 절약형)
1. 뼈대: `python3 tools/scaffold.py --conf db/schools/<학교>.yaml --id <ID> --year <학년도> --track '<계열>' --qno <n> --sub 1:10,2:15 --from '<출처 문구 p.N>' [--mock]`
   → 고정 필드(source·메타·rubric 규칙)는 손대지 말고 TODO 만 채운다.
2. 원문 옮기기: 먼저 `pdftotext <pdf> -f N -l N -` 로 한글을 가져오고, 수식·그림이 있는 줄만 이미지로 본다.
   부분 이미지: `python3 tools/crop.py --pdf <pdf> --find "<그 줄의 한글 몇 글자>" --out <scratch>/c.png` (쪽 전체 렌더 금지, 꼭 필요할 때만 -r 110).
3. 해설: 학교 공식 모범답안 흐름을 B 해설로(`source_flow: 모범답안 p.N`). draft 단계에서는 A 해설·hint_ladder·intent 생략 가능(경고만).
   채점: 공식 채점기준 배점 분할(없으면 `# 채점기준 미공개 — 단계별 추정`).
4. 검산: `verify/<PREFIX>.py` 를 tools/vlib.py 로 짧게 — 기대값은 반드시 `V.ans(no, sub[, key])`(하드코딩 금지).
   `python3 verify/<PREFIX>.py` 는 FAIL 줄과 SUMMARY 만 나온다(-v 로 전체).
5. 원본 대조: `python3 tools/source_check.py --src <원본.pdf> items/<학교>/PAST/<시험지 ID들>` →
   DIFF·PTS·NOALIGN·OUTSIDE 는 고치거나(내 오기) 크롭으로 확인 후 원본과 같으면 그대로. ORDER(분수 배치 등)는 크롭으로 수식만 확인.
6. 문항 YAML 을 다시 볼 때는 통째 Read 대신 `python3 tools/show.py <ID> [필드…]`.

## 핵심 기준(사용자)
- 문제는 원문 그대로: 제시문·발문·배점·단서·표·그림 설명을 한 글자씩. 모르는 글자는 그 자리에 `[판독확인: 이유]`. 원문 어미 그대로.
- 그림은 `[그림 n: 설명]`(번호는 원본과 같게, 설명은 콜론 뒤 — 대조기가 설명을 자동으로 뺀다).
- elements: materials·concepts·tools·conditions·stimuli_used·subpart_links.
- 공식 답이 틀렸다고 보이면 고치지 말고 보고(오류 수정은 독립 교차검증 규칙을 따른다).
- 공용 파일(db/, tools/, taxonomy, 카탈로그) 수정 금지. 오개념 후보는 proposals/<학교>_<yy>_catalog.yaml.
- 수식 규칙: db/제작지침.md §5(`$` 안쪽 공백 금지, `\(` 금지, 표 안 `|` 금지).

## 완료 조건 · 보고(5줄 이내)
`bash tools/precommit_check.sh <학교> [대조목록.tsv]` 가 a)~d) PASS.
보고: 만든 ID 범위 · 안 만든 문항과 이유 · precommit 결과 줄 · [판독확인] 위치 · 사람이 볼 것.
