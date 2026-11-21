#!/bin/bash
# KHU 전용 검증: 공용 카탈로그(db/*.yaml)를 고치지 않고, 임시 사본에 KHU 추가분(db/catalog_add_KHU/*.yaml)을 합쳐 validate.py 실행
# 사용: bash db/catalog_add_KHU/validate_khu.sh [items/KHU/PAST/KHU-G26-01.yaml ...]   (인자 없으면 KHU 전체)
set -e
SRC=$(cd "$(dirname "$0")/../.." && pwd)
TMP=$(mktemp -d)
cp -r $SRC/db $SRC/tools $SRC/items $TMP/ 2>/dev/null || true
[ -d $SRC/verify ] && cp -r $SRC/verify $TMP/
python3 - "$TMP" <<'P'
import sys, glob, yaml
T = sys.argv[1]
con, mis = [], []
for f in sorted(glob.glob(f"{T}/db/catalog_add_KHU/*.yaml")):
    d = yaml.safe_load(open(f, encoding="utf-8")) or {}
    con += d.get("concepts") or []; mis += d.get("misconceptions") or []
dump = lambda L: yaml.safe_dump(L, allow_unicode=True, sort_keys=False, width=200)
cp = yaml.safe_load(open(f"{T}/db/concepts.yaml", encoding="utf-8"))["concepts"]
have = {c["id"] for c in cp}; seen=set(); con2=[]
for c in con:
    if c["id"] in have or c["id"] in seen: continue
    seen.add(c["id"]); con2.append(c)
if con2: open(f"{T}/db/concepts.yaml", "a", encoding="utf-8").write("\n" + dump(con2))
ids=[m["id"] for m in mis]
dup={i for i in ids if ids.count(i)>1}
if dup: print("ERROR 오개념 id 중복:", dup)
if mis: open(f"{T}/db/misconceptions.yaml", "a", encoding="utf-8").write("\n" + dump(mis))
P
cd $TMP
if [ $# -eq 0 ]; then python3 tools/validate.py items/KHU; else python3 tools/validate.py "$@"; fi
rc=$?
rm -rf $TMP
exit $rc
