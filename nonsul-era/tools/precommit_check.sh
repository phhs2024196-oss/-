#!/bin/bash
# 학교 DB 자동 커밋 전 일괄 검사(운영 규칙: 통과 시 승인 없이 커밋·푸시). 출력은 PASS/FAIL 한 줄씩.
# 사용: bash tools/precommit_check.sh <학교코드> [원본대조목록.tsv]
#   원본대조목록.tsv: 한 줄에 "원본.pdf<TAB>items/<학교>/PAST/<ID묶음 glob>" (시험지 1장 = 1줄). 없으면 d) 생략 표시.
cd "$(dirname "$0")/.." || exit 2
S="$1"; LIST="$2"; fail=0
[ -z "$S" ] && { echo "사용: $0 <학교코드> [목록.tsv]"; exit 2; }
out=$(python3 tools/validate.py "items/$S" --quiet 2>&1 | tail -1)
echo "$out" | grep -q "errors=0 " && echo "PASS a) validate: $out" || { echo "FAIL a) validate: $out"; fail=1; }
v=$(bash tools/run_verify.sh $(ls verify/"$S"-*.py 2>/dev/null) 2>&1); rc=$?
[ $rc -eq 0 ] && echo "PASS b) verify: $(echo "$v" | grep -c 'rc=0')개 스크립트" || { echo "FAIL b) verify:"; echo "$v" | grep -E "FAIL|rc=[1-9]" | head -5; fail=1; }
bad=$( (git diff --name-only HEAD; git ls-files --others --exclude-standard) | grep -vE "^(nonsul-era/)?(items/$S/|verify/$S-|proposals/)" )
[ -z "$bad" ] && echo "PASS c) 변경 범위: 학교 파일만" || { echo "FAIL c) 학교 밖 변경:"; echo "$bad" | head -5; fail=1; }
if [ -n "$LIST" ]; then
  n=0; while IFS=$'\t' read -r pdf glob; do
    [ -z "$pdf" ] && continue
    r=$(python3 tools/source_check.py --src "$pdf" $glob 2>&1); rc=$?
    [ $rc -ne 0 ] && { n=$((n+1)); echo "     $(echo "$r" | tail -1) ← $pdf"; }
  done < "$LIST"
  [ $n -eq 0 ] && echo "PASS d) 원본 대조" || { echo "FAIL d) 원본 대조 확인 필요 시험지 $n장(남은 항목을 크롭으로 '원본과 같음' 확인하면 통과)"; fail=1; }
else
  echo "SKIP d) 원본 대조 목록 없음"
fi
exit $fail
