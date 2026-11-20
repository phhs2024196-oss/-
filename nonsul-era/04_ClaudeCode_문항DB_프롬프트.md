# Claude Code 문항 DB 만들기 — 프롬프트 모음 v1.0

작성 2026-11-20. Claude Code(웹·앱·CLI)에 **그대로 붙여 넣는** 프롬프트다. `〈 〉` 부분만 바꿔 쓴다.
규칙은 저장소의 `CLAUDE.md`·`db/제작지침.md`에 있으므로 프롬프트에는 "무엇을 할지"만 쓴다.

| # | 프롬프트 | 언제 | 예상 규모 |
|---|---|---|---|
| P0 | 저장소 셋업 + 300문항 이관 | 12월 첫 주, 1회 | 세션 1개 |
| P1 | 이관 300문항 정리(범위 태깅·범위 밖 풀이 수정·r1/r2) | P0 직후, 학교별 1회 | 학교당 세션 1~2개 |
| P2 | 유형 1세트(10문항) 신규 생산 ★반복 | 매일 | 유형당 세션 1개 |
| P3 | 기출·모의 DB 입력 | 학교별 1회 | 학교당 세션 1개 |
| P4 | 검토 모델 선정(오류 심은 20문항) | P2 본격 생산 전 1회 | 세션 1개 |
| P5 | 교재 조판(PDF) | 교재 출고 때 | 세션 1개 |
| P6 | 주간 점검 | 매주 일요일 | 20분 |

긴 작업은 프롬프트 맨 앞에 `/goal 〈완료 조건〉`을 붙이면 조건을 만족할 때까지 멈추지 않는다(예: `/goal SSU-T03 신규 10문항이 verified 이고 validate ERROR 0`).
서브에이전트 모델 지정(`model: fable / opus / sonnet`)은 쓰는 Claude Code 버전·요금제에서 지원되는지 처음 한 번 확인하고, 안 되면 "모델 지정 없이 별도 서브에이전트"로 바꿔 쓴다.

---

## P0. 저장소 셋업 + 300문항 이관

사전 준비(사람): GitHub에 비공개 저장소 `nonsul-era-db`를 만들고, 그 저장소로 Claude Code 세션을 연다. 세션에 `phhs2024196-oss/-` 저장소 읽기 권한이 필요하다.

```text
이 저장소를 「논술의 시대」 수리논술 문항 DB 저장소로 셋업해줘.

1. phhs2024196-oss/- 저장소의 브랜치 claude/eloquent-thompson-enlegs 에서 nonsul-era/ 폴더 전체(CLAUDE.md, db/, tools/, requirements.txt, .gitignore, 01~05 문서)를 이 저장소 루트로 복사해. 접근이 안 되면 멈추고 나에게 말해.
2. 같은 저장소의 브랜치 claude/confident-volta-323thl 의 nonsul/ 폴더(유형별 N제 300문항: soongsil·dongguk·sejong 의 Txx.md, verify_Txx.py)를 임시 폴더에 받아.
3. pip install -r requirements.txt 후 python3 tools/import_legacy.py <임시 nonsul 폴더> 를 실행해 items/, verify/, books/ 를 만들어.
4. python3 tools/validate.py --quiet → ERROR 0 확인. bash tools/run_verify.sh → 30개 파일 모두 10/10 PASS 확인. python3 tools/build_db.py --stats 실행.
5. 하나라도 실패하면 원인을 고치고(이관 스크립트 버그면 스크립트를, 문항 오류면 보고만) 다시 돌려. 문항 내용은 이 단계에서 고치지 마.
6. 원본 PDF(nonsul/pdf/*.pdf)는 sources/legacy_pdf/ 에 복사해.
7. 커밋 "Initialize question DB with 300 legacy items" 후 push.
보고: 이관 수, validate SUMMARY, verify 결과 30줄 요약, build --stats 의 학교·범위 태그 표.
```

---

## P1. 이관 300문항 정리 (학교별로 1회씩: SSU → DGU → SJU)

```text
〈SSU〉 이관 문항 100개(items/〈SSU〉/)를 새 검수 흐름에 올려줘. CLAUDE.md 와 db/제작지침.md 를 먼저 읽어.

A. 범위 태깅: 각 문항의 subjects 와 scope_2028 을 직접 판단해 확정하고 scope_confirmed: true 로 바꿔.
   기준: 문제를 푸는 데 꼭 필요한 가장 높은 과목. common(수학·수Ⅰ·수Ⅱ·확통만), calc2(미적분 필요), geom(기하 필요).
   모범답안이 범위 밖 도구를 써서 geom 으로 잡힌 것은 문제 자체 기준으로 다시 판단해.
   휴리스틱 추정과 다르게 바꾼 문항은 목록으로 보고해.
B. 범위 밖 풀이 수정: validate.py 가 "기하 도구 의심" 경고를 낸 문항(세종: J01-01·04·06·07, J10-03·04·05·08·09·10 = SJU-T01-001·004·006·007, SJU-T10-003·004·005·008·009·010)의 모범답안을
   그 학교 범위 안의 도구로 다시 써. 채점표 단계·배점은 모범답안과 1:1 로 맞추고, 정답은 바뀌면 안 돼.
C0. v2 완비(문항마다, 서브에이전트 model: opus 로 문항 몇 개씩): 원문 내용은 바꾸지 말고 ① 각 채점 마크에 mark_type(M/A/B/R) 지정·필요하면 depends_on·ft·not_penalized 추가(배점 합은 유지) ② 해설 A(방향·핵심) 작성 ③ legacy.wrong_answers 를 오개념으로 구조화: db/misconceptions.yaml 에 없으면 카탈로그에 먼저 추가하고 소문항 misconceptions 에 연결(outcome_code 70~78, wrong_work·detect_cue·feedback) ④ 처방 remedy_links(개념은 db/concepts.yaml 에 추가) ⑤ intent·idea_tags·behavior_domain 작성 ⑥ legacy.answer_text 로 정답이 빈 소문항 채우기 ⑦ depends_on 실제 관계로 확인. 끝나면 legacy 블록은 지우지 말고 둔다.
C. r1 검토: 유형별로 서브에이전트(model: sonnet)를 하나씩 띄워 db/제작지침.md §9 의 r1 항목을 검토하게 해. 결과를 각 문항 reviews 에 기록하고 통과 문항은 status: r1_pass.
D. r2 검토: r1_pass 문항마다 서로 독립인 서브에이전트 2개(model: opus)에게 제시문·발문만 주고 소문항별 최종 답만 받아. 둘 다 정답과 일치하면 reviews 에 r2_calc_1·r2_calc_2(verdict: pass, computed_answers 기록) 추가, status: r2_pass.
   불일치면 sympy 로 누가 맞는지 판정 → 문항 오류면 수정 후 다시 r2, 검토자 오류면 note 에 기록하고 그 검토만 1회 재실행.
E. verify: r2 까지 끝난 유형은 모두 run_verify.sh 재실행(이관 때의 legacy_verify 기록은 인정 안 됨). PASS + validate ERROR 0 이면 reviews 맨 끝에 stage: verify, result: pass 추가, status: verified.
F. 유형 하나 끝날 때마다 build_db.py 후 커밋·push.
보고: 유형별 verified 수, 범위 태그 변경 목록, 수정·폐기한 ID와 이유, 내가 직접 봐야 할 문항(증명 논리 등).
```

동국(DGU)은 2027까지 기하가 출제범위에 포함되므로(2026 논술가이드북) 모범답안의 기하 도구는 범위 안이다. 다만 2028 대비용 태그는 정확히 geom 으로 단다. 기하 유형(DGU-T11) 10문항은 2027학년도 기준에 따라 P2로 신규 제작한다.

---

## P2. 유형 1세트 신규 생산 ★ (매일 반복하는 기본 프롬프트)

```text
〈SSU〉 〈T03 접선·도형과 극한〉 유형에 신규 문항 〈10〉개를 만들어 verified 까지 올려줘. CLAUDE.md, db/제작지침.md, db/taxonomy.yaml 을 먼저 읽어.
조건: 2028 범위 태그는 〈common 우선(가능하면 10개 전부 common)〉. 난이도 분포 〈L4 2 / L5 6 / L6 2〉.

1. 중복 확인: python3 tools/build_db.py 후 python3 tools/search.py 〈핵심 단어〉 --school 〈SSU〉 와 python3 tools/search.py 〈핵심 단어〉(전 학교)로 기존 문항의 열쇠·설정 목록을 만들어. 새 문항은 이것들과 열쇠·설정이 겹치면 안 돼.
2. 설계(서브에이전트 model: fable): 기존 목록과 제작지침을 주고 10행 설계도(ID·L·형식·열쇠·정답 예상·막힘·대표 오답)를 받아. 형식 4종 이상·같은 형식 3회 이하 확인.
3. 제작(서브에이전트 model: opus, 문항 2~3개씩 나눠 병렬): 설계도 행을 주고 db/item_template.yaml(양식 v2) 형식의 문항 파일(해설 A·B, 1마크=1행 채점표, 오개념·처방 포함)과 검산 함수 p〈순번〉() 을 받아. ID 는 이 유형 마지막 순번 다음부터.
   검산 파일은 세트마다 새로: db/verify_template.py 를 verify/〈SSU〉-〈T03〉-s〈세트번호〉.py 로 복사해 PREFIX 를 바꾸고 함수를 넣어(세트 = 순번 10개 단위, 011~020 = s2). 기존 verify/〈SSU〉-〈T03〉.py 에 덧붙이지 마.
4. 검산: run_verify.sh 로 이 유형 실행. FAIL 이면 제작 서브에이전트에게 돌려보내 수정(정답이 지저분하면 설정 변경).
5. r1(서브에이전트 model: sonnet, 제작과 다른 에이전트): 제작지침 §9 r1 항목. fail → 그 문항은 제작부터 백지 재시작.
6. r2(서로 독립 서브에이전트 2개, model: opus): 제시문·발문만 주고 소문항별 최종 답만 받기 → reviews 에 r2_calc_1·r2_calc_2(computed_answers 필수). 비교는 네가 해. 불일치 → sympy 로 판정, 문항 오류면 restart_from: solution 으로 풀이부터 재작성.
   또 build/r2_packets/〈ID〉.md 에 제시문·논제만 담은 파일을 만들어 둬(내가 다른 계열 AI로 추가 확인할 수 있게).
7. 같은 문항 3회 연속 탈락 → retired, 새 ID 로 재제작하고 새 문항에 replaces: 〈폐기 ID〉.
8. 전부 통과하면 validate ERROR 0, run_verify PASS 확인 → reviews 에 verify pass 기록 → status: verified → validate 다시(검수 기록 검사) → build_db.py --stats → 커밋·push.
보고: CLAUDE.md 보고 형식 + 단계별 탈락 횟수(제작 FAIL, r1 fail, r2 불일치)와 대략의 소요 시간.
```

변형: 공통 교재는 `〈COM〉`과 공통 유형 코드(taxonomy.yaml에 먼저 추가), 기출 변형은 "각 문항 kind: variant, parent: 〈기출 ID〉, 원 기출의 열쇠를 바꿔서"를 조건에 추가.

---

## P3. 기출·모의 DB 입력

사전 준비(사람): 학교 입학처에서 받은 기출·모의·선행학습영향평가 보고서 PDF를 세션에 첨부하거나 `sources/〈SSU〉/`에 넣는다(파일명 `SSU_2026_실전.pdf`). PDF는 `.gitignore`로 커밋되지 않으므로 원본은 구글 드라이브에 보관하고, 클라우드 세션에서는 매번 첨부한다.

```text
sources/〈SSU〉/ 의 PDF 들로 〈SSU〉 기출 DB 를 만들어줘. 02_문항DB_저장계획서.md §4·§8 과 CLAUDE.md 를 따라.
1. 각 PDF 에서 문항을 하나씩 items/〈SSU〉/PAST/〈ID〉.md 로 만들어. ID: 실전 〈SSU〉-G연도2-번호2, 모의 〈SSU〉-M연도2-번호2 (회차 구분이 있으면 연도 뒤 A/B).
2. origin: past_exam/mock_exam, source: {rights_holder: 〈대학〉, license: all_rights_reserved, permission_status: not_requested, allowed_use: internal_only, derivative_level: verbatim, institution, admission_year, admission_type, track, question_no, official_url, retrieved_at, original_points, rights_notice(보고서 표지 문구 그대로)}.
3. 제시문·발문은 원문 그대로 옮기고(내부 분석용 — allowed_use: internal_only 라 교재·공개 JSONL 에서 자동 제외), 해설 A·B·채점 마크·오개념은 직접 작성. 대학 공개 채점 기준·예시답안은 legacy 대신 tags/메모가 아니라 solutions(ALT)·rubric 에 반영하고 출처 쪽수를 note 로.
4. 각 문항 정답을 sympy 로 검산: 연도·회차마다 db/verify_template.py 를 verify/〈SSU〉-G26.py(모의는 -M27) 로 복사해 PREFIX="〈SSU〉-G26", WIDTH=2 로 바꾸고 문항 번호 함수 p04 를 넣어. front matter verify: verify/〈SSU〉-G26.py::p04.
5. level·form·subjects·scope_2028 태깅, type 은 기존 유형 코드 중 가장 가까운 것(없으면 taxonomy.yaml 에 새 유형 제안만 하고 type: null).
6. PDF 판독이 불확실한 수식은 [판독확인] 표시를 메모에 남기고 나에게 목록으로 보고.
7. 검수: r1_scope(서브에이전트 model: sonnet — 원문 PDF와 옮긴 제시문·발문 대조 + 해설·채점표 검토) → r2_calc_1·r2_calc_2(제시문·발문만 주고 독립 풀이) → verify. 전부 pass 면 status: verified. 원문 수록 교재(기생기사·다다익선)는 대학 허락(permission_status: granted + permission_ref) 전까지 "출처 + 자체 해설" 판으로만 조판된다.
8. validate → verify → build → 커밋.
보고: 연도·회차별 문항 수, 유형 분포(어떤 유형이 몇 번 나왔는지), 판독확인 목록.
```

---

## P4. 검토 모델 선정 (오류 심은 20문항)

```text
r1·r2 검토 설정이 오류를 잘 잡는지 시험해줘.
1. verified 문항 20개를 골라 사본을 build/seeded/ 에 만들고, 각 사본에 오류 1개를 심어(정답 계산 실수 5, 조건 누락으로 답 비유일 4, 범위 밖 도구 3, 채점표-모범답안 불일치 4, 증명 논리 비약 4). 어떤 오류를 심었는지는 build/seeded/answer_key.yaml 에만 적어.
2. r1 설정(model: sonnet)과 r2 설정(model: opus, 제시문·논제만)으로 20개를 검토하게 해. 검토자에게 오류가 있다는 사실은 알리지 마.
3. 표로 보고: r1 단독 / r2 단독 / r1+r2 합산(둘 중 하나라도 잡으면 잡은 것)으로 잡은 개수.
   기준: 합산 18/20 이상, 그리고 r2 는 정답이 바뀌는 오류(계산 실수 5 + 조건 누락 4 = 9개) 중 8개 이상. r2 는 모범답안·채점표를 보지 않으므로 나머지 11개는 r1 몫이다.
   기준 미달이면 검토 지시문을 고쳐 다시 시험(최대 3회).
4. 최종 검토 지시문을 db/제작지침.md §9 아래에 "검토 지시문 v1" 으로 추가하고 커밋. build/seeded/ 는 커밋하지 마.
```

---

## P5. 교재 조판

```text
books/〈SSU-YUYU-2028〉.yaml 의 문항으로 A4 인쇄용 PDF 를 만들어줘.
1. python3 tools/validate.py --book books/〈SSU-YUYU-2028〉.yaml --strict 가 통과하는지 먼저 확인(수록 문항 전부 verified 이상, 경고 0). 아니면 멈추고 목록 보고.
2. 기존 조판 도구(phhs2024196-oss/- 브랜치 claude/confident-volta-323thl 의 nonsul/print/: build_print.py, print.css, prerender_print.js, print_pdf.js)를 tools/print/ 로 가져와, 입력을 books yaml + items md 로 바꿔.
   구성: 표지 → 빠른 정답 → 유형별 풀이 팁 → 문제(홀수 쪽)/해설(짝수 쪽).
3. source.allowed_use 가 full_reproduction 이 아닌 문항(허락 없는 기출)은 원문 대신 "출처(대학·학년도·번호) + 자체 해설"만 나오게. validate --book 이 원문 수록을 오류로 막는다.
4. 표지에 판(edition)·발행일, 마지막 쪽에 "오류 신고: 〈스토어 문의〉 · 정오표: 〈주소〉" 를 넣어.
5. build/pdf/ 에 저장하고 쪽수, 수식 렌더 실패 건수를 보고. 실패 0 이 될 때까지 고쳐.
6. 출고가 확정되면 python3 tools/build_db.py --lock books/〈SSU-YUYU-2028〉.yaml 로 스냅샷(.lock.yaml)을 만들어 커밋해. 이후 build 때마다 바뀐 문항이 "정오표 대상"으로 보고된다.
```

---

## P6. 주간 점검 (일요일)

```text
이번 주 문항 DB 점검 보고서를 써줘.
1. git log --since="7 days ago" 요약, build_db.py --stats.
2. 학교·유형별 verified 수와 목표(〈파일럿 3개교 716문항, 2027-03-31〉) 대비 진도율, 이 속도로 목표일까지 끝나는지.
3. 이번 주 단계별 탈락률(reviews 의 fail 비율). 30% 넘는 단계가 있으면 원인 추정과 지침 수정안.
4. scope_confirmed: false 남은 수, retired 수, errata open 수.
5. reports/weekly/〈YYYY-MM-DD〉.md 로 저장하고 커밋(build/ 는 커밋되지 않으므로 쓰지 말 것).
```

---

## 자주 생기는 문제

| 증상 | 조치 |
|---|---|
| 사용량 한도에 걸림 | 유형 1개 단위로 끊어 커밋했으므로 다음 세션에서 "SSU-T03 이어서" 로 재개. 루프 3회 제한이 무한 소모를 막는다 |
| r2 가 계속 불일치 | 문항이 모호할 가능성이 높다. r1 에 "조건만으로 답이 유일한가"를 다시 시키고, 그래도면 retired |
| 검산 스크립트가 정답을 하드코딩 | r1 체크 항목. 발견 시 그 유형 전체 검산 재작성 |
| 같은 열쇠 반복 | P2 1단계 중복 조회 누락. tools/search.py 결과를 설계 서브에이전트에 반드시 전달 |
| 판매 후 오류 신고 | errata/〈날짜〉-〈ID〉.yaml 작성(item_id, reported, channel, description, severity, status) → 문항 수정·version+1 → build 의 "정오표 대상" 목록으로 정오표 작성 |
