# 논술의 시대 — 문항 DB 저장소 규칙

수리논술 교재·AI 첨삭용 문항 DB. 원본은 `items/**/*.md`(문항 1개 = 파일 1개), DB(`build/nonsul.db`)는 파생물이다.

## 반드시 읽을 것
- `db/제작지침.md` — 난이도 L, 형식 12종, 학교별 규격, 금지목록, 검산 규칙, 게이트 G1~G7, 채점 원칙, 검토 규칙
- `db/taxonomy.yaml` — 학교·유형·분류값의 유일한 기준. 새 값은 여기에 먼저 추가
- `db/item_template.md` — 새 문항 뼈대
- `02_문항DB_저장계획서.md` — ID 체계, 상태 흐름, 저작권 원칙

## 명령
```bash
pip install -r requirements.txt
python3 tools/validate.py [경로]          # ERROR 0 이어야 커밋
python3 tools/validate.py --book books/X.yaml --strict   # 교재 출고 직전
bash tools/run_verify.sh [verify/X.py]     # sympy 검산
python3 tools/build_db.py --stats          # DB 빌드 + 진도·분포 통계 + 출고 후 변경(정오표 대상) 보고
python3 tools/build_db.py --lock books/X.yaml   # 출고 스냅샷(.lock.yaml 커밋)
python3 tools/search.py 단어 [--school SSU --type T03]   # 중복 확인(부분 문자열)
```

## 절대 규칙
1. ID는 바꾸거나 재사용하지 않는다. 새 ID = 그 유형 마지막 순번 + 1.
2. `build/`의 DB를 직접 고치지 않는다. 항상 md 수정 → build.
3. `status`는 draft → r1_pass → r2_pass → verified 순서로만 올린다. 각 단계 기록을 `reviews`에 남긴다. r1·r2a·r2b·verify 의 마지막 기록이 모두 `result: pass`가 아니면 verified 금지(검사기가 막음).
4. 자기가 만든 문항을 같은 컨텍스트에서 검토하지 않는다. r1·r2는 서브에이전트(또는 별도 세션)로, r2 검토자에게는 제시문·논제만 준다.
5. 검산 스크립트에 정답을 하드코딩해 자기 자신과 비교하지 않는다. 새 세트는 `db/verify_template.py`를 복사해 `verify/<학교>-<유형>-s<세트>.py`로 만든다(기존 세트 1 파일에 함수를 덧붙이지 않는다 — 고정 목록이라 실행되지 않음).
6. 모범답안도 그 학교 출제범위 안의 도구만 쓴다(세종·숭실에서 벡터·내적 금지 등).
7. 기출 원문은 내부 분석용으로만 저장. `copyright.original_text: false`면 교재·사이트에 원문을 싣지 않는다.
8. 학생 개인정보·API 키·기출 원본 PDF(`sources/**/*.pdf`, gitignore)를 저장소에 넣지 않는다.
8-1. 출고한 교재의 문항을 고치면 `version`+1, `errata/<날짜>-<ID>.yaml` 작성(정오표 원본).
9. 같은 문항 3회 연속 탈락 → `retired` 후 새 ID로 재제작하고 새 문항에 `replaces: <폐기 ID>`를 적는다. 루프를 무한히 돌리지 않는다.
9-1. verify 기록은 마지막 r1·r2 기록 뒤에 있어야 한다(검토 중 고쳤으면 검산 재실행). 이관 문항의 옛 검산은 `legacy_verify`로만 인정되지 않는다.
10. 작업 단위(유형 1개 등)가 끝날 때마다 validate → verify → build → 커밋. 커밋 메시지: `[SSU-T03] 011~020 verified (r2 불일치 1건 수정)` 형식.

## 보고 형식 (작업 끝 메시지, 10줄 이내)
처리 문항 ID 범위 · 상태별 개수 · validate/verify SUMMARY 줄 그대로 · 탈락·수정한 ID와 이유 · 폐기 ID · 사람이 확인할 것(범위 판단, 증명 논리 등 수치로 확인 못 한 것).
