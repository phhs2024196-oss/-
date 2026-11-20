#!/usr/bin/env python3
"""items/**/*.md + books/*.yaml + db/taxonomy.yaml → build/nonsul.db (SQLite).

사용: python3 tools/build_db.py [--db build/nonsul.db] [--stats]
- DB 는 매번 새로 만든다(파생물). 원본은 md 파일이다.
- validate.py 에서 ERROR 가 있으면 빌드하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import subprocess
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import KIT, content_hash, load_item, load_taxonomy, parse_rubric  # noqa: E402


def j(v):
    return json.dumps(v, ensure_ascii=False) if v not in (None, "") else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=pathlib.Path, default=KIT / "build" / "nonsul.db")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    if not args.skip_validate:
        r = subprocess.run([sys.executable, str(KIT / "tools" / "validate.py"), "--quiet"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print("validate.py ERROR — 빌드 중단")
            return 1

    tax = load_taxonomy()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()
    con = sqlite3.connect(args.db)
    con.executescript((KIT / "db" / "schema.sql").read_text(encoding="utf-8"))

    for code, s in tax["schools"].items():
        con.execute("INSERT INTO schools VALUES (?,?,?,?,?,?,?)",
                    (code, s["name"], int(bool(s.get("pilot"))), int(bool(s.get("candidate"))), s.get("format"),
                     j(s.get("scope_2027")), j(s.get("scope_2028"))))
    for sch, types in tax["types"].items():
        for t, name in (types or {}).items():
            con.execute("INSERT INTO types VALUES (?,?,?)", (sch, t, name))

    files = sorted((KIT / "items").rglob("*.md"))
    rows = []
    for f in files:
        meta, sec, body = load_item(f)
        rows.append((meta, sec, body, f))
    # parent 가 먼저 들어가도록: 부모 없는 문항 → 변형 순
    rows.sort(key=lambda r: (r[0].get("parent") is not None, r[0]["id"]))
    for meta, sec, body, f in rows:
        rec = {
            "id": meta["id"], "legacy_id": meta.get("legacy_id"), "kind": meta["kind"], "school": meta["school"],
            "type": meta.get("type"), "parent": meta.get("parent"), "level": meta.get("level"),
            "form": meta.get("form"), "points": meta["points"], "sub_points": j(meta.get("sub_points") or None),
            "subjects": j(meta.get("subjects")), "scope_2028": meta.get("scope_2028", "unknown"),
            "scope_confirmed": int(bool(meta.get("scope_confirmed"))), "stuck": meta.get("stuck"),
            "concepts": j(meta.get("concepts")), "key_idea": meta.get("key_idea"), "answer": meta.get("answer"),
            "status": meta["status"], "verify_ref": meta.get("verify"), "source": j(meta.get("source")),
            "original_text": int(bool((meta.get("copyright") or {}).get("original_text"))),
            "version": meta.get("version", 1), "created": str(meta.get("created")),
            "updated": str(meta.get("updated")), "path": str(f.relative_to(KIT)), "body_md": body,
            "passage_md": sec.get("제시문"), "prompt_md": sec.get("논제"), "solution_md": sec.get("모범답안"),
            "rubric_md": sec.get("채점표"), "errors_md": sec.get("대표 오답"), "content_hash": content_hash(body),
        }
        con.execute(f"INSERT INTO items ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", tuple(rec.values()))
        for r in parse_rubric(sec.get("채점표", "")):
            con.execute("INSERT OR REPLACE INTO rubric_steps VALUES (?,?,?,?,?)",
                        (meta["id"], r["step"], r["sub_question"], r["content"], r["points"]))
        for n, rv in enumerate(meta.get("reviews") or [], 1):
            con.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?)",
                        (meta["id"], n, rv.get("stage"), rv.get("reviewer"), str(rv.get("date")), rv.get("result"),
                         rv.get("note")))
        con.execute("INSERT INTO items_fts VALUES (?,?,?,?)",
                    (meta["id"], body, meta.get("key_idea") or "", " ".join(meta.get("concepts") or [])))

    for bf in sorted((KIT / "books").glob("*.yaml")):
        b = yaml.safe_load(bf.read_text(encoding="utf-8"))
        con.execute("INSERT INTO books VALUES (?,?,?,?,?,?)",
                    (b["code"], b["title"], b["series"], b.get("school"), str(b.get("season")), b.get("edition", 1)))
        for it in b.get("items", []):
            con.execute("INSERT INTO placements VALUES (?,?,?,?)", (b["code"], it["id"], it.get("chapter"), it["ord"]))
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"built {args.db.relative_to(KIT) if args.db.is_relative_to(KIT) else args.db}: items={n}, "
          f"rubric_steps={con.execute('SELECT COUNT(*) FROM rubric_steps').fetchone()[0]}, "
          f"books={con.execute('SELECT COUNT(*) FROM books').fetchone()[0]}")
    if args.stats:
        print("\n[학교·상태]")
        for r in con.execute("SELECT school, status, COUNT(*) FROM items GROUP BY 1,2 ORDER BY 1,2"):
            print("  ", *r)
        print("\n[학교·2028 범위 태그]")
        for r in con.execute("SELECT school, scope_2028, COUNT(*) FROM items GROUP BY 1,2 ORDER BY 1,2"):
            print("  ", *r)
        print("\n[유형 요약] school type n L4 L5 L6 forms publishable common calc2 geom unknown")
        for r in con.execute("SELECT * FROM v_type_summary ORDER BY school, type"):
            print("  ", *r)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
