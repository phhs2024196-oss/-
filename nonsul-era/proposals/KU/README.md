# 고려대(KU) 입력에 필요한 공용 파일 변경 제안 (2026-11-21)

공용 파일(db/)은 학교 스레드가 직접 고치지 않기로 해서, 필요한 변경을 여기에 모았다. 병합 담당이 적용한다.

1. `taxonomy.patch` — schools.KU 에 format·scope_2027·note 추가, types.KU T01~T06 신설. 적용: `git apply nonsul-era/proposals/KU/taxonomy.patch`
2. `misconceptions_add.yaml` — MC-*-7101~7220 번호대 64개. `db/misconceptions.yaml` 의 misconceptions 목록 끝에 덧붙인다.
3. `concepts_add.yaml` — 새 개념 64개(루트 CON-ALG·CON-GEO·CON-PRB·CON-CNT 포함). 다른 학교 브랜치가 같은 ID(특히 루트)를 먼저 넣었으면 그 항목은 건너뛴다.

세 가지를 적용한 상태에서 `python3 tools/validate.py items/KU` → files=24 errors=0 warnings=4(KU-M25B scope_2028 추정값), `bash tools/run_verify.sh verify/KU-*.py` → 6파일 24/24 PASS 확인.
적용 전에는 KU 유형·오개념·개념 미등록으로 validate 가 ERROR 를 낸다.
