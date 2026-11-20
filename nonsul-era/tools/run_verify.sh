#!/bin/bash
# 전 유형 sympy 검산 일괄 실행: bash tools/run_verify.sh [verify/SSU-T01.py ...]
# 각 스크립트는 마지막 줄에 "SUMMARY Txx: n/n PASS" 를 출력하고, FAIL 이 있으면 종료코드 1.
cd "$(dirname "$0")/../verify" || exit 1
files=("$@"); [ ${#files[@]} -eq 0 ] && files=(*.py)
fail=0
for f in "${files[@]}"; do
  f=$(basename "$f")
  out=$(timeout 900 python3 "$f" 2>&1); rc=$?
  echo "$f: rc=$rc  $(echo "$out" | grep SUMMARY | tail -1)"
  if [ $rc -ne 0 ]; then fail=1; echo "$out" | grep -E "FAIL|Error|Traceback" | head -5; fi
done
exit $fail
