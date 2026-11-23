# 교재 참고 DB (TB)

시중 수리논술 교재 18종(21개 파일, 약 2,370쪽)의 문항·해설·개념 요약을 모은 내부 참고 DB. 학교별 기출 DB(db/)와 별개.

- **원문 데이터는 이 리포에 없다.** 리포가 공개라서 타사 교재 원문(raw/, items/, notes/, all.jsonl)은 비공개 프로젝트 폴더 `textbook-db/`에만 둔다(사용자 결정 2026-11-23).
- 리포에는 형식(BRIEF.md)·병합 도구(tools/build.py)·현황(stats.md)·분석 문서(ANALYSIS.md)만 둔다.
- 사용 원칙: 구조·아이디어 참고용. 타사 문항·해설 문장을 우리 교재에 그대로 싣지 않는다.

## 구조(프로젝트 폴더)
- `raw/<BOOK>/pAAA-BBB.yaml` 쪽 구간별 입력 원본(BRIEF.md 형식: id·book·page·chapter·section·role·label·cited_source·stem·points·answer·solution_summary·subject·unit_path·concepts·skills·level·flags)
- `notes/<BOOK>/pAAA-BBB.md` 구간별 개념·양식 요약
- `items/<BOOK>.yaml`, `all.jsonl` — `python3 tools/build.py` 결과(책 뒤 해설·해설편을 label로 문항에 병합, solution_ref·merged_into 기록)

## 검증(2026-11-23)
- 표본 86문항(교재별 10%) 독립 검증자 4명이 원본 쪽 이미지와 대조: 발문 85/86 일치(1건 누락 구절 수정), 답 의심 3건은 원본 표기 유지 + `tools/review_flags.yaml` 로 표시.
