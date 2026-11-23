# 원본 대조 검증 지시 (독립 검증자)
너는 교재 DB 입력이 원본과 같은지 검사한다. 한국어. 파일 수정 금지(읽기만), git 금지.
입력: 검사 목록 JSON(각 항목 id·book·page·stem·answer·points). 원본 쪽 이미지: /tmp/claude-0/tb/<book>/p-<page>.png (자릿수는 폴더의 파일명 그대로: ls 로 확인). 문항이 다음 쪽으로 이어지면 다음 쪽도 본다. 작은 글씨는 pdftoppm -r 200 으로 그 쪽만 다시 렌더해 crop(PIL) 해서 본다(원본 PDF 경로는 /tmp/claude-0/tb/books.tsv 에서 코드→파일명, 폴더 /mnt/project-files/).
각 항목: 원본 이미지와 stem을 대조(발문·조건·수식·숫자·소문항·배점). answer가 있으면 원본(같은 쪽 또는 해설)에 맞는지 가능한 범위에서 확인(없으면 '답확인불가').
판정: PASS(실질 차이 없음, 띄어쓰기·LaTeX 표기 차이는 무시) / DIFF(수식·숫자·단어 누락·오기 — 정확히 어디가 어떻게 다른지) / 판정불가(이유).
결과를 /tmp/claude-0/tb/verify_out_<n>.tsv 로 저장: id<TAB>판정<TAB>내용(한 줄).
보고: PASS/DIFF/판정불가 개수 한 줄 + DIFF 목록(id: 차이) 만.
