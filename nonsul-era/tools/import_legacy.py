#!/usr/bin/env python3
"""기존 '유형별 수리논술 N제' 300문항(nonsul/<school>/Txx.md + verify_Txx.py)을
양식 v2(문항 1개 = YAML 1개, items/<SCH>/<Txx>/<ID>.yaml)로 옮긴다.

사용:
  python3 tools/import_legacy.py <기존 nonsul 폴더> [--out <kit 폴더>] [--dry-run]

원칙
- 원본 문항 내용(제시문·논제·모범답안·정답·채점표·대표 오답)은 글자를 바꾸지 않고 v2 구조로 나눠 담기만 한다.
- 원본에서 끌어낼 수 없는 v2 정보(마크 종류 M/A/B/R, 해설 A, 오개념 카탈로그 연결, 출제의도, 처방 링크)는
  지어내지 않는다. 비워 두고(mark_type: null 등) P1 정리 작업에서 채운다. validate.py 는 draft 에서는 경고,
  verified 부터는 오류로 막는다.
- 상태는 draft. 기존 sympy 검산 통과는 reviews 에 legacy_verify 로만 남긴다(verified 근거로 인정 안 됨).
- scope_2028 은 키워드 휴리스틱 추정값(scope_confirmed: false).
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import KIT, dump_yaml, guess_scope_2028, load_taxonomy  # noqa: E402

LEGACY = {"soongsil": ("S", "SSU"), "dongguk": ("D", "DGU"), "sejong": ("J", "SJU")}
MARKERS = ["제시문", "논제", "모범답안", "정답", "채점표", "대표 오답", "열쇠"]
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def clean(s: str) -> str:
    """줄 끝 공백 제거(YAML | 블록으로 깔끔하게 저장되도록)."""
    return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())


def parse_design_table(md: str) -> dict[str, dict]:
    out = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and re.match(r"^[SDJ]\d\d-\d\d$", cells[0]):
            out[cells[0]] = {"key": cells[3], "stuck": cells[5], "wrong": cells[6]}
    return out


def split_blocks(md: str, prefix: str) -> list[tuple[str, str]]:
    parts = re.split(rf"^###\s+({prefix}\d\d-\d\d)\s*$", md, flags=re.M)
    return [(parts[i], re.sub(r"\n---\s*$", "", parts[i + 1].strip()).strip()) for i in range(1, len(parts), 2)]


def parse_block(text: str) -> dict:
    meta = re.search(r"^`(L\d)\s*·\s*([^·`]+?)\s*·\s*(\d+)점`", text, re.M)
    if not meta:
        raise ValueError("메타 줄(`L4 · 추론 · 25점`) 없음")
    pieces = re.compile(r"\*\*(" + "|".join(MARKERS) + r")\*\*").split(text[meta.end():])
    sec = {pieces[i]: pieces[i + 1].strip() for i in range(1, len(pieces), 2)}
    return {"level": meta.group(1), "form": meta.group(2).strip(), "points": int(meta.group(3)), **sec}


def parse_rubric_rows(table: str) -> list[list[str]]:
    rows = []
    for line in table.splitlines():
        s = line.strip()
        if s.startswith("|") and not re.match(r"^\|[\s:|-]*$", s):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells[-1] != "배점":
                rows.append(cells)
    return rows


def split_prompt(prompt: str, total: int) -> tuple[str, list[dict]]:
    """논제 → (공통 발문, [{id, stem, points}]). 소문항이 없으면 소문항 1개."""
    lines = prompt.splitlines()
    starts = [i for i, ln in enumerate(lines) if re.match(r"^\((\d)\)\s", ln.strip())]
    if not starts:
        return "", [{"id": "1", "stem": clean(prompt), "points": total}]
    subs, tail = [], []
    bounds = starts + [len(lines)]
    for k in range(len(starts)):
        chunk = lines[bounds[k]:bounds[k + 1]]
        # 마지막 소문항 뒤의 '<15줄 이내> [30점]' 같은 전체 안내 줄은 공통 발문으로 옮긴다
        while k == len(starts) - 1 and chunk and (
                not chunk[-1].strip() or re.match(r"^<.*>\s*([\[(]\d+점[\])])?\s*$", chunk[-1].strip())):
            last = chunk.pop().strip()
            if last:
                tail.insert(0, last)
        text = "\n".join(chunk).strip()
        ms = [m for m in re.finditer(r"\s*[\[(](\d+)점[\])]", text) if outside_math(text, m.start())]
        m = ms[-1] if ms else None             # 배점 표시는 줄 끝이 아니라 별행 수식 앞에 있기도 함
        pts = int(m.group(1)) if m else None
        if m:
            text = (text[:m.start()] + text[m.end():]).strip()
        text = re.sub(r"^\((\d)\)\s*", "", text)
        subs.append({"id": str(k + 1), "stem": clean(text), "points": pts})
    tail = [re.sub(r"\s*[\[(]\d+점[\])]\s*$", "", t) for t in tail]
    stem = clean("\n".join(lines[:starts[0]] + ([""] + tail if tail else [])))
    return stem, subs


def outside_math(text: str, pos: int) -> bool:
    return text[:pos].replace("\\$", "").count("$") % 2 == 0


def split_numbered(text: str, n: int) -> list:
    """'(1) … (2) … (3) …' 를 수식 밖에서, 1→2→3 순서로 나오는 표시만 경계로 삼아 나눈다($f(2)$ 같은 것은 무시)."""
    cuts, prev = [], 0                       # (번호, 시작, 끝). 증명 소문항은 정답에서 빠지기도 하므로 번호 건너뜀 허용
    for m in re.finditer(r"\((\d)\)", text):
        num = int(m.group(1))
        if prev < num <= n and outside_math(text, m.start()):
            cuts.append((num, m.start(), m.end()))
            prev = num
    out = [None] * n
    for k, (num, _, end) in enumerate(cuts):
        out[num - 1] = clean(text[end:(cuts[k + 1][1] if k + 1 < len(cuts) else len(text))])
    return out


def split_answer(ans: str, n: int) -> list:
    return [clean(ans)] if n == 1 else split_numbered(ans, n)


def split_solution(sol: str) -> list[tuple[str, str]]:
    """모범답안 → [(①, 내용)]. 단계 표시(①②…)는 줄 처음이든 문장 중간이든 수식 밖이면 경계로 본다.
    첫 표시 앞의 글은 첫 단계에 붙이고, 단계 앞뒤의 소문항 표시 '(2)' 는 지운다."""
    idx = [(m.start(), m.group(1)) for m in re.finditer(rf"([{CIRCLED}])", sol) if outside_math(sol, m.start())]
    if not idx:
        return [("①", clean(sol))]
    out = []
    for k, (pos, mark) in enumerate(idx):
        end = idx[k + 1][0] if k + 1 < len(idx) else len(sol)
        body = sol[pos:end].strip()[1:].strip()
        if k == 0 and re.sub(r"^\(\d\)\s*", "", sol[:pos].strip()):
            body = sol[:pos].strip() + "\n\n" + body
        body = re.sub(r"\s*\(\d\)\s*$", "", body)          # 다음 소문항 표시 '(2)' 가 끝에 붙은 경우
        out.append((mark, clean(body)))
    return out


def to_int(s: str) -> int:
    d = re.sub(r"[^\d]", "", s)
    return int(d) if d else 0


def build_item(sch: str, tcode: str, lid: str, b: dict, d: dict) -> dict:
    seq = int(lid.split("-")[1])
    iid = f"{sch}-{tcode}-{seq:03d}"
    stem, subs = split_prompt(b.get("논제", ""), b["points"])
    if len(subs) > 1 and (any(s["points"] is None for s in subs) or sum(s["points"] for s in subs) != b["points"]):
        # 소문항 배점을 원문에서 읽을 수 없으면(숭실처럼 논제 하나에 (1)(2)) 지어내지 않고 소문항 1개로 둔다
        stem, subs = "", [{"id": "1", "stem": clean(b.get("논제", "")), "points": b["points"]}]
    if len(subs) == 1:
        stem, subs[0]["stem"] = subs[0]["stem"], ""      # 단일 논제: 발문은 공통 발문에, 소문항 발문은 비움
    answers = split_answer(b.get("정답", ""), len(subs))
    rows = parse_rubric_rows(b.get("채점표", ""))
    steps = dict(split_solution(b.get("모범답안", "")))
    # 채점표 행 → 소문항 배정(4열이면 소문항 열, 아니면 소문항 배점 누적으로)
    bounds, acc = [], 0
    for s in subs:
        acc += s["points"] or 0
        bounds.append(acc)
    acc, k = 0, 0
    per_sub: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        mark = r[0]
        lead = re.match(r"^\((\d)\)\s*(.*)$", mark)          # '(1)①' 처럼 단계 칸에 소문항이 붙은 경우
        if len(r) >= 4 and re.match(r"^\(\d\)$", r[1]):
            sid, content = r[1].strip("()"), " | ".join(r[2:-1])
        elif lead and len(subs) > 1:
            sid, mark, content = lead.group(1), lead.group(2), " | ".join(r[1:-1])
        else:
            sid, content = (str(k + 1) if len(subs) > 1 else "1"), " | ".join(r[1:-1])
        pts = to_int(r[-1])
        per_sub[sid].append((mark, content, pts))
        acc += pts
        while k < len(bounds) - 1 and acc >= bounds[k]:
            k += 1
    scope, subjects = guess_scope_2028(" ".join([b.get("제시문", ""), b.get("논제", ""), b.get("모범답안", "")]))
    subparts = []
    for i, s in enumerate(subs):
        sid = s["id"]
        crit, sol_steps = [], []
        for n, (mark, content, pts) in enumerate(per_sub.get(sid, []), 1):
            num = CIRCLED.index(mark) + 1 if mark in CIRCLED else n
            stid, cid = f"s{num}", f"{sid}.S{num}"
            crit.append({"id": cid, "mark_type": None, "points": pts, "evidence": clean(content), "steps": [stid]})
            if mark in steps:
                sol_steps.append({"id": stid, "content": steps[mark], "criteria": [cid]})
        is_proof = bool(re.search(r"보이시오|증명하시오", s["stem"] or stem))
        subparts.append({
            "id": sid, "stem": s["stem"], "points": s["points"] if s["points"] is not None else b["points"],
            # G6(동국·세종: 마지막 소문항이 앞 결과 사용)에 따른 기본값 — P1 에서 실제 의존 관계로 확인
            "depends_on": [str(j) for j in range(1, i + 1)] if i and i == len(subs) - 1 and sch in ("DGU", "SJU") else [],
            "answer": {"kind": "proof" if is_proof else "value", "value": answers[i], "equivalents": [],
                       "tolerance": None},
            "solutions": [{"id": f"S{sid}-B", "audience": "B", "method_no": 1, "steps": sol_steps}],
            "rubric": {"criteria": crit, "deductions": []},
            "misconceptions": [],
        })
    stimuli = [{"label": "제시문", "body": clean(b["제시문"])}] if b.get("제시문", "").strip() else []
    proofy = any(sp["answer"]["kind"] == "proof" for sp in subparts)
    return {
        "schema_version": "2.0", "id": iid, "legacy_id": lid, "version": 1, "status": "draft",
        "exam_family": "math_nonsul", "item_type": "proof" if proofy else "derivation", "origin": "original",
        "school": sch, "type": tcode, "variant_of": None, "replaces": None,
        "level": b["level"], "form": b["form"], "points": b["points"], "time_estimate_min": None,
        "intent": "", "key_idea": clean(b.get("열쇠", d.get("key", ""))), "stuck": d.get("stuck", ""),
        "idea_tags": [], "tags": [],
        "curriculum": {"version": "2015", "subjects": subjects, "scope_2028": scope, "scope_confirmed": False,
                       "standards": [], "behavior_domain": None, "unit_path": [], "key_concepts": []},
        "difficulty": {"rater": "author", "p_value": None, "n": 0},
        "source": {"rights_holder": "논술의 시대", "license": "own_original", "permission_status": "not_needed",
                   "allowed_use": "full_reproduction", "derivative_level": "idea_only"},
        "verify": f"verify/{sch}-{tcode}.py::p{seq:02d}",
        "stimuli": stimuli, "stem": stem, "subparts": subparts, "remedy_links": [],
        "legacy": {"answer_text": clean(b.get("정답", "")), "wrong_answers": clean(b.get("대표 오답", "")),
                   "design_wrong": d.get("wrong", ""),
                   "note": "유형별 수리논술 N제(2026-11-17판)에서 이관. 마크 종류·해설 A·오개념 연결·depends_on 확인은 P1에서"},
        "reviews": [{"stage": "legacy_verify", "actor_type": "system", "actor_id": f"sympy(legacy verify_{tcode}.py)",
                     "item_version": 1, "verdict": "pass", "date": "2026-11-17",
                     "note": "기존 N제 자체 검산 통과. 새 파이프라인 r1·r2·verify 미실시"}],
        "created": "2026-11-17", "updated": "2026-11-20",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=KIT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tax = load_taxonomy()
    stats, scope_by_school, errors = collections.Counter(), collections.defaultdict(collections.Counter), []
    for folder, (prefix, sch) in LEGACY.items():
        books = []
        for tfile in sorted((args.legacy_dir / folder).glob("T[0-9][0-9].md")):
            tcode = tfile.stem
            if tcode not in tax["types"][sch]:
                errors.append(f"{tfile}: taxonomy 에 {sch}/{tcode} 없음")
                continue
            md = tfile.read_text(encoding="utf-8")
            design = parse_design_table(md)
            for lid, text in split_blocks(md, prefix):
                try:
                    item = build_item(sch, tcode, lid, parse_block(text), design.get(lid, {}))
                except (ValueError, KeyError, IndexError) as e:
                    errors.append(f"{lid}: {e}")
                    continue
                stats[sch] += 1
                scope_by_school[sch][item["curriculum"]["scope_2028"]] += 1
                books.append(item["id"])
                if not args.dry_run:
                    p = args.out / "items" / sch / tcode / f"{item['id']}.yaml"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(dump_yaml(item), encoding="utf-8")
            vsrc = args.legacy_dir / folder / f"verify_{tcode}.py"
            if vsrc.exists() and not args.dry_run:
                (args.out / "verify").mkdir(exist_ok=True)
                shutil.copy(vsrc, args.out / "verify" / f"{sch}-{tcode}.py")
        if books and not args.dry_run:
            bk = {"code": f"{sch}-YUYU-2028", "title": f"{tax['schools'][sch]['name']} 유유상종 (유형별 N제) 0판",
                  "series": "유유상종", "school": sch, "season": "2028", "edition": 0,
                  "items": [{"id": i, "chapter": i.split("-")[1], "ord": n + 1} for n, i in enumerate(books)]}
            (args.out / "books").mkdir(exist_ok=True)
            (args.out / "books" / f"{sch}-YUYU-2028.yaml").write_text(dump_yaml(bk), encoding="utf-8")
    print("이관 문항 수:", dict(stats), "합계", sum(stats.values()))
    for sch, c in scope_by_school.items():
        print(f"  {sch} scope_2028 추정: {dict(c)}")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
