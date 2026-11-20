# 논술의 시대 — 사업 문서 + 문항 DB 키트

수리논술 교재·기출 DB·AI 첨삭 사업 「논술의 시대」의 기준 문서와, 문항 DB를 바로 시작할 수 있는 도구 모음. (2026-11-20)

## 문서 (읽는 순서)

| # | 문서 | 무엇 | 누가 쓰나 |
|---|---|---|---|
| 05 | [실행 로드맵](05_실행_로드맵.md) | **지금부터 날짜별로 할 일**, 결정 게이트, 문의 메일 양식 | 대표 |
| 01 | [사업계획서](01_사업계획서.md) | 시장·상품·자산·일정·손익분기·수익 추정·리스크·법규 | 대표, claude.ai 프로젝트 |
| 02 | [문항DB 저장계획서](02_문항DB_저장계획서.md) | 저장 구조, ID 체계, 문항 형식, 2028 범위 태그, 검수 흐름, 저작권, 300문항 이관 시험 결과 | 대표, Claude Code |
| 03 | [프로젝트 지침](03_프로젝트_지침.md) | claude.ai 프로젝트 「논술의 시대」에 붙여 넣을 지침 + 지식 파일 목록 + 기존 정보 정리 절차 | claude.ai |
| 04 | [Claude Code 프롬프트](04_ClaudeCode_문항DB_프롬프트.md) | 셋업(P0)·이관 정리(P1)·유형 생산(P2)·기출 입력(P3)·검토 모델 선정(P4)·조판(P5)·주간 점검(P6) | Claude Code |

## 문항 DB 키트

```
CLAUDE.md            Claude Code 작업 규칙(이 폴더를 저장소 루트로 쓰면 자동 로딩)
db/taxonomy.yaml     학교·유형·난이도·형식·과목·상태 기준표
db/schema.sql        SQLite 스키마(파생 DB)
db/item_template.md  문항 1개 파일 뼈대
db/제작지침.md        문항 제작·검산·검토 규칙(기존 N제 지침 승계)
db/verify_template.py 새 세트 검산 파일 뼈대(p숫자 함수 자동 실행)
tools/import_legacy.py  기존 300문항 → 문항별 파일 이관
tools/validate.py       검사기(ID·분류값·배점 합·소문항 소계·단계 중복·수식 표기·범위 밖 도구·세트 분포·검수 기록·교재 참조)
tools/build_db.py       md → build/nonsul.db + 통계 + 출고 후 변경(정오표 대상) 보고, --lock 으로 출고 스냅샷
tools/search.py         부분 문자열 검색(중복 확인)
tools/run_verify.sh     sympy 검산 일괄 실행
items/ verify/ books/ errata/ reports/   (P0 실행 후 채워짐)
```

빠른 시작:
```bash
pip install -r requirements.txt
python3 tools/import_legacy.py <기존 nonsul 폴더>   # 기존 300문항 이관
python3 tools/validate.py --quiet
bash tools/run_verify.sh
python3 tools/build_db.py --stats
```

기존 300문항 원본은 이 저장소의 브랜치 `claude/confident-volta-323thl`의 `nonsul/`에 있다.
