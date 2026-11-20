#!/usr/bin/env python3
"""items/**/*.md + books/*.yaml + errata/*.yaml + db/taxonomy.yaml → build/nonsul.db (SQLite).

사용:
  python3 tools/build_db.py [--stats]               # DB 재생성(+통계)
  python3 tools/build_db.py --lock books/X.yaml     # 교재 출고 시점 스냅샷 books/X.lock.yaml 작성(커밋할 것)
- DB 는 매번 새로 만드는 파생물이다. 원본은 md·yaml 파일이다.
- validate.py 에 ERROR 가 있으면 빌드하지 않는다.
- 출고 스냅샷(.lock.yaml)과 현재 문항 해시가 다르면 "정오표 대상"으로 보고한다.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sqlite3
import subprocess
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import KIT, content_hash, load_books, load_item, load_taxonomy, parse_rubric  # noqa: E402


def j(v):
    if v is None or v == "" or v == []:
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def topo_order(rows: list) -> list:
    """parent 가 먼저 들어가도록 정렬(변형의 변형도 처리)."""
    by_id = {r[0]["id"]: r for r in rows}
    out, done = [], set()

    def visit(iid: str, stack: tuple = ()):
        if iid in done or iid not in by_id:
            return
        if iid in stack:
            raise SystemExit(f"parent 순환 참조: {' → '.join(stack + (iid,))}")
        parent = by_id[iid][0].get("parent")
        if parent:
            visit(parent, stack + (iid,))
        done.add(iid)
        out.append(by_id[iid])

    for iid in sorted(by_id):
        visit(iid)
    return out


def lock_book(book_path: pathlib.Path) -> int:
    b = yaml.safe_load(book_path.read_text(encoding="utf-8"))
    files = {p.stem: p for p in (KIT / "items").rglob("*.md")}
    entries = []
    for it in b.get("items", []):
        f = files[it["id"]]
        meta, _, _ = load_item(f)
        entries.append({"id": it["id"], "version": meta.get("version", 1),
                        "hash": content_hash(f.read_text(encoding="utf-8"))})
    out = book_path.with_name(book_path.stem + ".lock.yaml")
    out.write_text(yaml.safe_dump({"book": b["code"], "edition": b.get("edition", 1),
                                   "locked": datetime.date.today().isoformat(), "items": entries},
                                  allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"locked {len(entries)} items → {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=pathlib.Path, default=KIT / "build" / "nonsul.db")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--lock", type=pathlib.Path)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    if not args.skip_validate:
        r = subprocess.run([sys.executable, str(KIT / "tools" / "validate.py"), "--quiet"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print("validate.py ERROR — 중단 (python3 tools/validate.py 로 상세 확인)")
            return 1
    if args.lock:
        return lock_book(args.lock)

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

    rows = []
    for f in sorted((KIT / "items").rglob("*.md")):
        meta, sec, body = load_item(f)
        rows.append((meta, sec, body, f))
    for meta, sec, body, f in topo_order(rows):
        rec = {
            "id": meta["id"], "legacy_id": meta.get("legacy_id"), "kind": meta["kind"], "school": meta["school"],
            "type": meta.get("type"), "parent": meta.get("parent"), "level": meta.get("level"),
            "form": meta.get("form"), "points": meta["points"], "sub_points": j(meta.get("sub_points")),
            "subjects": j(meta.get("subjects")), "scope_2028": meta.get("scope_2028", "unknown"),
            "scope_confirmed": int(bool(meta.get("scope_confirmed"))), "stuck": meta.get("stuck"),
            "concepts": j(meta.get("concepts")), "key_idea": meta.get("key_idea"), "answer": meta.get("answer"),
            "status": meta["status"], "verify_ref": meta.get("verify"), "source": j(meta.get("source")),
            "original_text": int(bool((meta.get("copyright") or {}).get("original_text"))),
            "version": meta.get("version", 1), "created": str(meta.get("created")),
            "updated": str(meta.get("updated")), "path": str(f.relative_to(KIT)), "body_md": body,
            "passage_md": sec.get("제시문"), "prompt_md": sec.get("논제"), "solution_md": sec.get("모범답안"),
            "rubric_md": sec.get("채점표"), "errors_md": sec.get("대표 오답"),
            "content_hash": content_hash(f.read_text(encoding="utf-8")),
        }
        con.execute(f"INSERT INTO items ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", tuple(rec.values()))
        rub, _ = parse_rubric(sec.get("채점표", ""))
        for r in rub:
            con.execute("INSERT INTO rubric_steps VALUES (?,?,?,?,?,?)",
                        (meta["id"], r["seq"], r["step"], r["sub_question"], r["content"], r["points"]))
        for n, rv in enumerate(meta.get("reviews") or [], 1):
            con.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?,?)",
                        (meta["id"], n, rv.get("stage"), rv.get("reviewer"), str(rv.get("date")), rv.get("result"),
                         rv.get("note")))
        con.execute("INSERT INTO items_fts VALUES (?,?,?,?)",
                    (meta["id"], body, meta.get("key_idea") or "", " ".join(meta.get("concepts") or [])))

    for bf, b in load_books():
        con.execute("INSERT INTO books VALUES (?,?,?,?,?,?)",
                    (b["code"], b["title"], b["series"], b.get("school"), str(b.get("season")), b.get("edition", 1)))
        for it in b.get("items", []):
            con.execute("INSERT INTO placements VALUES (?,?,?,?)", (b["code"], it["id"], it.get("chapter"), it["ord"]))
        lock = bf.with_name(bf.stem + ".lock.yaml")
        if lock.exists():
            lk = yaml.safe_load(lock.read_text(encoding="utf-8"))
            for e in lk.get("items", []):
                con.execute("INSERT INTO book_locks VALUES (?,?,?,?,?)",
                            (b["code"], e["id"], e["version"], e["hash"], str(lk.get("locked"))))
    for ef in sorted((KIT / "errata").glob("*.yaml")) if (KIT / "errata").exists() else []:
        e = yaml.safe_load(ef.read_text(encoding="utf-8"))
        con.execute("INSERT INTO errata VALUES (?,?,?,?,?,?,?,?)",
                    (ef.stem, e["item_id"], str(e["reported"]), e.get("channel"), e["description"],
                     e.get("severity"), e.get("fixed_version"), e.get("status", "open")))
    con.commit()

    def q(sql: str) -> int:
        return con.execute(sql).fetchone()[0]

    n_items, n_steps = q("SELECT COUNT(*) FROM items"), q("SELECT COUNT(*) FROM rubric_steps")
    n_books = q("SELECT COUNT(*) FROM books")
    n_errata = q("SELECT COUNT(*) FROM errata WHERE status IN ('open','confirmed')")
    where = args.db.relative_to(KIT) if args.db.is_relative_to(KIT) else args.db
    print(f"built {where}: items={n_items}, rubric_steps={n_steps}, books={n_books}, errata_open={n_errata}")
    changed = con.execute("""SELECT l.book, l.item_id, l.version, i.version FROM book_locks l
                             JOIN items i ON i.id = l.item_id WHERE l.content_hash <> i.content_hash""").fetchall()
    if changed:
        print(f"\n[출고 후 변경된 문항 = 정오표 대상 {len(changed)}건]")
        for b, iid, v0, v1 in changed:
            flag = "" if v1 > v0 else "  ← version 을 올리지 않음"
            print(f"   {b} {iid} v{v0}→v{v1}{flag}")
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
