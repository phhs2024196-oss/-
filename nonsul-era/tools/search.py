#!/usr/bin/env python3
"""문항 부분 문자열 검색(조사가 붙은 한국어 단어도 찾음). 중복 확인용.

사용: python3 tools/search.py 접선 [극한 ...] [--school SSU] [--type T03] [--field key_idea|body]
- 여러 단어는 AND. 기본은 열쇠(key_idea)·개념·본문 전체에서 찾는다.
- build/nonsul.db 가 필요하다(python3 tools/build_db.py).
"""
import argparse
import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import KIT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("words", nargs="+")
ap.add_argument("--school")
ap.add_argument("--type")
ap.add_argument("--field", choices=["all", "key_idea", "body"], default="all")
ap.add_argument("--limit", type=int, default=50)
a = ap.parse_args()
db = KIT / "build" / "nonsul.db"
if not db.exists():
    sys.exit("build/nonsul.db 없음 — 먼저 python3 tools/build_db.py")
col = {"all": "(coalesce(key_idea,'') || ' ' || coalesce(concepts,'') || ' ' || body_md)",
       "key_idea": "coalesce(key_idea,'')", "body": "body_md"}[a.field]
where, params = [], []
for w in a.words:
    where.append(f"instr({col}, ?) > 0")
    params.append(w)
if a.school:
    where.append("school = ?"); params.append(a.school)
if a.type:
    where.append("type = ?"); params.append(a.type)
con = sqlite3.connect(db)
rows = con.execute(f"SELECT id, status, level, form, key_idea FROM items WHERE {' AND '.join(where)} "
                   f"ORDER BY id LIMIT ?", params + [a.limit]).fetchall()
for r in rows:
    print(f"{r[0]}  {r[1]:9} {r[2]} {r[3]:5} {(r[4] or '')[:70]}")
print(f"-- {len(rows)}건" + (" (limit 도달)" if len(rows) == a.limit else ""))
