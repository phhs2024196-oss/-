#!/usr/bin/env python3
"""items/**/*.yaml + db/*.yaml + books/ + errata/ → build/nonsul.db (SQLite) + build/items.jsonl (교환 표준)
+ build/items_public.jsonl (외부 공개 가능분만).

사용:
  python3 tools/build_db.py [--stats]               # 재생성(+통계)
  python3 tools/build_db.py --lock books/X.yaml     # 출고 스냅샷 books/X.lock.yaml (validate --book --strict 통과 시)
- validate.py 에 ERROR 가 있으면 빌드하지 않는다.
- 중복 탐지: 완전 중복(정규화 해시) / 유사 중복(숫자·변수 마스킹 MinHash, Jaccard ≥ 0.8) / 발상 중복(idea_tags).
- 공개용 JSONL 은 status ≥ verified 이고 source.allowed_use = full_reproduction 인 문항만 담는다(저작권 차단).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import sqlite3
import subprocess
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import (KIT, MINHASH_PERM, dump_yaml, item_content_hash, item_files, item_texts, load_books,  # noqa: E402
                 load_catalog, load_item, load_taxonomy, minhash, normalize)

NEAR_THRESHOLD = 0.8
BANDS, ROWS = 32, 4                      # 32 × 4 = 128 (LSH)


def j(v):
    if v is None or v == "" or v == [] or v == {}:
        return None
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)


def topo_order(items: list[dict]) -> list[dict]:
    by_id = {it["id"]: it for it in items}
    out, done = [], set()

    def visit(iid, stack=()):
        if iid in done or iid not in by_id:
            return
        if iid in stack:
            raise SystemExit(f"variant_of/replaces 순환 참조: {' → '.join(stack + (iid,))}")
        for ref in (by_id[iid].get("variant_of"), by_id[iid].get("replaces")):
            if ref:
                visit(ref, stack + (iid,))
        done.add(iid)
        out.append(by_id[iid])

    for iid in sorted(by_id):
        visit(iid)
    return out


def lock_book(book_path: pathlib.Path) -> int:
    r = subprocess.run([sys.executable, str(KIT / "tools" / "validate.py"), "--book", str(book_path), "--strict",
                        "--quiet"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print("이 교재는 validate --book --strict 를 통과하지 못해 출고 스냅샷을 만들 수 없음")
        return 1
    b = yaml.safe_load(book_path.read_text(encoding="utf-8"))
    files = {p.stem: p for p in item_files()}
    entries = []
    for x in b.get("items", []):
        it = load_item(files[x["id"]])
        entries.append({"id": x["id"], "version": it.get("version", 1), "hash": item_content_hash(it)})
    out = book_path.with_name(book_path.stem + ".lock.yaml")
    out.write_text(dump_yaml({"book": b["code"], "edition": b.get("edition", 1),
                              "locked": datetime.date.today().isoformat(), "items": entries}), encoding="utf-8")
    print(f"locked {len(entries)} items → {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=pathlib.Path, default=KIT / "build" / "nonsul.db")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--lock", type=pathlib.Path)
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    if args.lock:
        return lock_book(args.lock)
    if not args.skip_validate:
        r = subprocess.run([sys.executable, str(KIT / "tools" / "validate.py"), "--quiet"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print("validate.py ERROR — 중단 (python3 tools/validate.py 로 상세 확인)")
            return 1

    tax = load_taxonomy()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()
    con = sqlite3.connect(args.db)
    con.executescript((KIT / "db" / "schema.sql").read_text(encoding="utf-8"))
    ins = lambda table, rec: con.execute(  # noqa: E731
        f"INSERT INTO {table} ({','.join(rec)}) VALUES ({','.join('?' * len(rec))})", tuple(rec.values()))

    for code, s in tax["schools"].items():
        ins("schools", {"code": code, "name": s["name"], "pilot": int(bool(s.get("pilot"))),
                        "candidate": int(bool(s.get("candidate"))), "format": s.get("format"),
                        "scope_2027": j(s.get("scope_2027")), "scope_2028": j(s.get("scope_2028"))})
    for sch, types in tax["types"].items():
        for t, name in (types or {}).items():
            ins("types", {"school": sch, "code": t, "name": name})
    for c in load_catalog("concepts").values():
        ins("concepts", {"id": c["id"], "name": c["name"], "parent": None, "curriculum_code": c.get("curriculum_code")})
    for c in load_catalog("concepts").values():
        if c.get("parent"):
            con.execute("UPDATE concepts SET parent=? WHERE id=?", (c["parent"], c["id"]))
        for pre in c.get("precedes") or []:
            ins("concept_edges", {"src": pre, "dst": c["id"], "type": "precedes"})
    for m in load_catalog("misconceptions").values():
        ins("misconceptions", {"id": m["id"], "name": m["name"], "concepts": j(m.get("concepts")),
                               "retired": int(bool(m.get("retired")))})

    files = item_files()
    items = [load_item(f) | {"__path": str(f.relative_to(KIT))} for f in files]
    jsonl, public, sigs = [], [], {}
    for it in topo_order(items):
        path = it.pop("__path")
        cur, src, txt = it.get("curriculum") or {}, it.get("source") or {}, item_texts(it)
        ins("items", {
            "id": it["id"], "legacy_id": it.get("legacy_id"), "version": it.get("version", 1),
            "schema_version": str(it.get("schema_version")), "status": it["status"], "exam_family": it["exam_family"],
            "item_type": it["item_type"], "origin": it["origin"], "school": it["school"], "type": it.get("type"),
            "variant_of": it.get("variant_of"), "replaces": it.get("replaces"), "level": it.get("level"),
            "form": it.get("form"), "points": it["points"], "time_estimate_min": it.get("time_estimate_min"),
            "intent": it.get("intent"), "key_idea": it.get("key_idea"), "stuck": it.get("stuck"),
            "idea_tags": j(it.get("idea_tags")), "tags": j(it.get("tags")),
            "curriculum_version": str(cur.get("version")), "subjects": j(cur.get("subjects")),
            "scope_2028": cur.get("scope_2028", "unknown"), "scope_confirmed": int(bool(cur.get("scope_confirmed"))),
            "behavior_domain": cur.get("behavior_domain"), "unit_path": j(cur.get("unit_path")),
            "key_concepts": j(cur.get("key_concepts")), "standards": j(cur.get("standards")),
            "difficulty": j(it.get("difficulty")), "rights_holder": src.get("rights_holder"),
            "license": src.get("license"), "permission_status": src.get("permission_status"),
            "allowed_use": src.get("allowed_use"), "derivative_level": src.get("derivative_level"), "source": j(src),
            "verify_ref": it.get("verify"), "stimuli": j(it.get("stimuli")), "stem": it.get("stem"),
            "path": path, "search_text": " ".join([it.get("key_idea") or "", " ".join(cur.get("key_concepts") or []),
                                                   txt["prompt"], txt["solution"], txt["rubric"]]),
            "content_hash": item_content_hash(it), "created": str(it.get("created")), "updated": str(it.get("updated")),
            "item_json": json.dumps(it, ensure_ascii=False, default=str)})
        for o, sp in enumerate(it.get("subparts") or [], 1):
            sid, ans = str(sp["id"]), sp.get("answer") or {}
            ins("subparts", {"item_id": it["id"], "sid": sid, "ord": o, "stem": sp.get("stem"), "points": sp["points"],
                             "depends_on": j(sp.get("depends_on")), "answer_kind": ans.get("kind"),
                             "answer_value": ans.get("value"), "answer": j(ans)})
            for so in sp.get("solutions") or []:
                ins("solutions", {"item_id": it["id"], "solution_id": so["id"], "sid": sid, "audience": so["audience"],
                                  "method_no": so.get("method_no"), "key_idea": so.get("key_idea"),
                                  "hint_ladder": j(so.get("hint_ladder"))})
                for k, st in enumerate(so.get("steps") or [], 1):
                    ins("solution_steps", {"item_id": it["id"], "solution_id": so["id"], "step_id": st["id"], "ord": k,
                                           "content": st.get("content"), "criteria": j(st.get("criteria"))})
            rub = sp.get("rubric") or {}
            if rub.get("rules"):
                ins("rubric_rules", {"item_id": it["id"], "sid": sid, "rules": j(rub["rules"])})
            for k, c in enumerate(rub.get("criteria") or [], 1):
                ins("rubric_criteria", {
                    "item_id": it["id"], "criterion_id": str(c["id"]), "sid": sid, "ord": k,
                    "mark_type": c.get("mark_type"), "points": c["points"], "evidence": c["evidence"],
                    "depends_on": j(c.get("depends_on")), "implied_ok": int(bool(c.get("implied_ok"))),
                    "ft": j(c.get("ft")), "accept": j(c.get("accept")), "cao": int(bool(c.get("cao"))),
                    "not_penalized": j(c.get("not_penalized")), "steps": j(c.get("steps"))})
            for d in rub.get("deductions") or []:
                ins("rubric_deductions", {"item_id": it["id"], "sid": sid, "code": d["code"], "condition": d.get("condition"),
                                          "effect": d.get("effect"), "value": d.get("value")})
            for mc in sp.get("misconceptions") or []:
                ins("item_misconceptions", {"item_id": it["id"], "sid": sid, "misconception_id": mc["id"],
                                            "outcome_code": str(mc["outcome_code"]), "wrong_work": mc.get("wrong_work"),
                                            "detect_cue": mc.get("detect_cue"), "feedback": mc.get("feedback"),
                                            "remedy": j(mc.get("remedy"))})
        for ln in it.get("remedy_links") or []:
            tr, tg = ln.get("trigger") or {}, ln.get("target") or {}
            ins("remedy_links", {"item_id": it["id"], "link_id": ln["id"], "trigger_type": tr.get("type"),
                                 "trigger_ref": None if tr.get("ref") is None else str(tr.get("ref")),
                                 "relation": ln.get("relation"), "target_kind": tg.get("kind"),
                                 "target_ref": tg.get("ref"), "priority": ln.get("priority")})
        for n, rv in enumerate(it.get("reviews") or [], 1):
            ins("reviews", {"item_id": it["id"], "seq": n, "stage": rv.get("stage"), "actor_type": rv.get("actor_type"),
                            "actor_id": rv.get("actor_id"), "prompt_version": rv.get("prompt_version"),
                            "item_version": rv.get("item_version"), "verdict": rv.get("verdict"),
                            "reason_codes": j(rv.get("reason_codes")), "restart_from": rv.get("restart_from"),
                            "computed_answers": j(rv.get("computed_answers")), "date": str(rv.get("date")),
                            "note": rv.get("note")})
        # 지문
        sig = minhash(txt["prompt"])
        sigs[it["id"]] = sig
        idea_key = hashlib.sha256("|".join(sorted(it.get("idea_tags") or [])).encode()).hexdigest()[:16] \
            if it.get("idea_tags") else None
        ins("fingerprints", {"item_id": it["id"], "norm_sha256": hashlib.sha256(normalize(txt["prompt"]).encode()).hexdigest(),
                             "idea_key": idea_key, "minhash": ",".join(map(str, sig.tolist()))})
        line = json.dumps(it, ensure_ascii=False, default=str)
        jsonl.append(line)
        if it["status"] in ("verified", "sampled", "published") and src.get("allowed_use") == "full_reproduction":
            public.append(line)

    # ── 중복 탐지 ──
    pairs = set()
    for kind, col in (("exact", "norm_sha256"), ("idea", "idea_key")):
        groups = collections.defaultdict(list)
        for iid, key in con.execute(f"SELECT item_id, {col} FROM fingerprints WHERE {col} IS NOT NULL"):
            groups[key].append(iid)
        for g in groups.values():
            for x in range(len(g)):
                for y in range(x + 1, len(g)):
                    pairs.add((min(g[x], g[y]), max(g[x], g[y]), kind, 1.0 if kind == "exact" else None))
    buckets = collections.defaultdict(list)
    for iid, sig in sigs.items():
        for band in range(BANDS):
            buckets[(band, tuple(sig[band * ROWS:(band + 1) * ROWS].tolist()))].append(iid)
    cand = {(min(a, b), max(a, b)) for g in buckets.values() if len(g) > 1 for i, a in enumerate(g) for b in g[i + 1:]}
    for a, b in cand:
        jac = float((sigs[a] == sigs[b]).sum()) / MINHASH_PERM
        if jac >= NEAR_THRESHOLD:
            pairs.add((a, b, "near", round(jac, 3)))
    for a, b, kind, jac in pairs:
        ins("dedup_pairs", {"a": a, "b": b, "kind": kind, "jaccard": jac})

    for bf, b in load_books():
        ins("books", {"code": b["code"], "title": b["title"], "series": b["series"], "school": b.get("school"),
                      "season": str(b.get("season")), "edition": b.get("edition", 1)})
        for x in b.get("items", []):
            ins("placements", {"book": b["code"], "item_id": x["id"], "chapter": x.get("chapter"), "ord": x["ord"],
                               "role": x.get("role")})
        lock = bf.with_name(bf.stem + ".lock.yaml")
        if lock.exists():
            lk = yaml.safe_load(lock.read_text(encoding="utf-8"))
            for e in lk.get("items", []):
                ins("book_locks", {"book": b["code"], "item_id": e["id"], "version": e["version"],
                                   "content_hash": e["hash"], "locked": str(lk.get("locked"))})
    for ef in sorted((KIT / "errata").glob("*.yaml")) if (KIT / "errata").exists() else []:
        e = yaml.safe_load(ef.read_text(encoding="utf-8"))
        ins("errata", {"id": ef.stem, "item_id": e["item_id"], "reported": str(e["reported"]), "channel": e.get("channel"),
                       "description": e["description"], "severity": e.get("severity"),
                       "fixed_version": e.get("fixed_version"), "status": e.get("status", "open")})
    con.commit()
    (args.db.parent / "items.jsonl").write_text("\n".join(jsonl) + ("\n" if jsonl else ""), encoding="utf-8")
    (args.db.parent / "items_public.jsonl").write_text("\n".join(public) + ("\n" if public else ""), encoding="utf-8")

    def q(sql):
        return con.execute(sql).fetchone()[0]

    where = args.db.relative_to(KIT) if args.db.is_relative_to(KIT) else args.db
    n_errata = q("SELECT COUNT(*) FROM errata WHERE status IN ('open','confirmed')")
    print(f"built {where}: items={q('SELECT COUNT(*) FROM items')}, subparts={q('SELECT COUNT(*) FROM subparts')}, "
          f"marks={q('SELECT COUNT(*) FROM rubric_criteria')}, books={q('SELECT COUNT(*) FROM books')}, "
          f"errata_open={n_errata}")
    print(f"jsonl: build/items.jsonl {len(jsonl)}줄, build/items_public.jsonl {len(public)}줄(공개 가능분)")
    dup = con.execute("SELECT kind, COUNT(*) FROM dedup_pairs GROUP BY kind").fetchall()
    print("중복 후보:", dict(dup) if dup else "없음")
    changed = con.execute("""SELECT l.book, l.item_id, l.version, i.version FROM book_locks l
                             JOIN items i ON i.id = l.item_id WHERE l.content_hash <> i.content_hash""").fetchall()
    if changed:
        print(f"\n[출고 후 변경된 문항 = 정오표 대상 {len(changed)}건]")
        for b, iid, v0, v1 in changed:
            print(f"   {b} {iid} v{v0}→v{v1}" + ("" if v1 > v0 else "  ← version 을 올리지 않음"))
    if args.stats:
        print("\n[학교·상태]")
        for r in con.execute("SELECT school, status, COUNT(*) FROM items GROUP BY 1,2 ORDER BY 1,2"):
            print("  ", *r)
        print("\n[학교·2028 범위 태그]")
        for r in con.execute("SELECT school, scope_2028, COUNT(*) FROM items GROUP BY 1,2 ORDER BY 1,2"):
            print("  ", *r)
        print("\n[v2 완비율] school items 마크종류지정비율 해설A수 오개념연결수")
        for r in con.execute("SELECT * FROM v_v2_completeness"):
            print("  ", r[0], r[1], f"{(r[2] or 0):.0%}", r[3], r[4])
        print("\n[유형 요약] school type n L4 L5 L6 forms publishable common calc2 geom unknown")
        for r in con.execute("SELECT * FROM v_type_summary ORDER BY school, type"):
            print("  ", *r)
        for r in con.execute("SELECT a, b, kind, jaccard FROM dedup_pairs ORDER BY kind, jaccard DESC LIMIT 20"):
            print("   중복후보", *r)
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
