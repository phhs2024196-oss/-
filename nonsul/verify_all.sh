#!/bin/bash
# 전 유형 검산 일괄 실행: bash verify_all.sh
cd "$(dirname "$0")"; fail=0
for d in soongsil dongguk sejong; do
  for f in $d/verify_T*.py; do
    [ -f "$f" ] || continue
    out=$( (cd $d && timeout 900 python3 "$(basename $f)") 2>&1 ); rc=$?
    echo "$d/$(basename $f): rc=$rc  $(echo "$out" | grep SUMMARY | tail -1)"
    [ $rc -ne 0 ] && fail=1 && echo "$out" | grep -E "FAIL|Error|Traceback" | head -5
  done
done
exit $fail
