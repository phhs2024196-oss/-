#!/usr/bin/env python3
"""기존 '유형별 수리논술 N제' 300문항(nonsul/<school>/Txx.md + verify_Txx.py)을
문항 1개 = 파일 1개 형식(items/<SCH>/<Txx>/<ID>.md)으로 옮긴다.

사용:
  python3 tools/import_legacy.py <기존 nonsul 폴더> [--out <kit 폴더>] [--dry-run]

- 원본 문항 내용은 한 글자도 바꾸지 않는다(제시문·논제·모범답안·정답·채점표·대표 오답 그대로).
- 상태는 draft 로 들여오고, 기존 sympy 검산 통과 사실은 reviews 에 기록한다.
  (새 파이프라인의 1차·2차 검토를 거쳐야 verified 로 올라간다.)
- scope_2028 은 키워드 휴리스틱 추정값이며 scope_confirmed: false 로 표시한다.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import shutil
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import KIT, guess_scope_2028, load_taxonomy  # noqa: E402

LEGACY = {"soongsil": ("S", "SSU"), "dongguk": ("D", "DGU"), "sejong": ("J", "SJU")}
MARKERS = ["제시문", "논제", "모범답안", "정답", "채점표", "대표 오답", "열쇠"]


def parse_design_table(md: str) -> dict[str, dict]:
    """유형 파일의 설계도 표 → {legacy_id: {열쇠, 막힘, ...}}"""
    out = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 7 and re.match(r"^[SDJ]\d\d-\d\d$", cells[0]):
            out[cells[0]] = {"level": cells[1], "form": cells[2], "key": cells[3],
                             "answer": cells[4], "stuck": cells[5], "wrong": cells[6]}
    return out


def split_blocks(md: str, prefix: str) -> list[tuple[str, str]]:
    parts = re.split(rf"^###\s+({prefix}\d\d-\d\d)\s*$", md, flags=re.M)
    blocks = []
    for i in range(1, len(parts), 2):
        text = parts[i + 1]
        text = re.sub(r"\n---\s*$", "", text.strip()).strip()
        blocks.append((parts[i], text))
    return blocks


def parse_block(text: str) -> dict:
    meta = re.search(r"^`(L\d)\s*·\s*([^·`]+?)\s*·\s*(\d+)점`", text, re.M)
    if not meta:
        raise ValueError("메타 줄(`L4 · 추론 · 25점`) 없음")
    pat = re.compile(r"\*\*(" + "|".join(MARKERS) + r")\*\*")
    pieces = pat.split(text[meta.end():])
    sec = {}
    for i in range(1, len(pieces), 2):
        sec[pieces[i]] = pieces[i + 1].strip()
    return {"level": meta.group(1), "form": meta.group(2).strip(), "points": int(meta.group(3)), **sec}


def sub_points_of(prompt: str, total: int) -> list[int]:
    """소문항 배점 [8점]/(80점) 목록. 문항 전체 배점 표기(예: '<15줄 이내> [30점]')는 뺀다."""
    nums = [int(n) for n in re.findall(r"[\[(](\d+)점[\])]", prompt)]
    if len(nums) > 1 and total in nums and sum(nums) - total == total:
        nums.remove(total)
    return nums


def to_rubric4(table: str, sub_pts: list[int]) -> str:
    """3열 채점표를 그대로 두되, 소문항이 있으면 배점 누적으로 소문항 열을 채운 4열 표로 바꾼다.
    누적합이 소문항 경계와 정확히 맞지 않으면 원본 3열 표를 그대로 둔다(validate 가 경고)."""
    rows = []
    for line in table.splitlines():
        s = line.strip()
        if s.startswith("|") and not re.match(r"^\|\s*-", s) and "배점" not in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            rows.append(cells)
    if not sub_pts or not rows or any(len(r) != 3 for r in rows):
        return table.strip()
    bounds, acc = [], 0
    for p in sub_pts:
        acc += p
        bounds.append(acc)
    out = ["| 단계 | 소문항 | 내용 | 배점 |", "|---|---|---|---|"]
    acc, k = 0, 0
    for step, content, pts in rows:
        digits = re.sub(r"[^\d]", "", pts)
        if not digits:
            return table.strip()
        acc += int(digits)
        if k >= len(bounds):
            return table.strip()
        out.append(f"| {step} | ({k + 1}) | {content} | {pts} |")
        if acc == bounds[k]:
            k += 1
        elif acc > bounds[k]:
            return table.strip()
    return "\n".join(out) if k == len(bounds) else table.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_dir", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=KIT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tax = load_taxonomy()
    stats = collections.Counter()
    scope_by_school = collections.defaultdict(collections.Counter)
    errors = []

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
                    b = parse_block(text)
                except ValueError as e:
                    errors.append(f"{lid}: {e}")
                    continue
                seq = int(lid.split("-")[1])
                iid = f"{sch}-{tcode}-{seq:03d}"
                prompt = b.get("논제", "")
                sub_pts = sub_points_of(prompt, b["points"])
                scope, subjects = guess_scope_2028(" ".join([b.get("제시문", ""), prompt, b.get("모범답안", "")]))
                d = design.get(lid, {})
                fm = {
                    "id": iid, "legacy_id": lid, "kind": "original", "school": sch, "type": tcode, "parent": None,
                    "level": b["level"], "form": b["form"], "points": b["points"],
                    "sub_points": sub_pts if len(sub_pts) > 1 else [],
                    "subjects": subjects, "scope_2028": scope, "scope_confirmed": False,
                    "concepts": [], "key_idea": b.get("열쇠", d.get("key", "")).strip(),
                    "stuck": d.get("stuck", ""), "answer": b.get("정답", "").strip(),
                    "source": None, "copyright": {"original_text": False},
                    "status": "draft", "verify": f"verify/{sch}-{tcode}.py::p{seq:02d}",
                    "reviews": [{"stage": "verify", "reviewer": "sympy(legacy verify_" + tcode + ".py)",
                                 "date": "2026-11-17", "result": "pass",
                                 "note": "기존 N제 제작 시 자체 검산 통과. 새 파이프라인 r1·r2 미실시"}],
                    "version": 1, "created": "2026-11-17", "updated": "2026-11-20",
                }
                rubric = to_rubric4(b.get("채점표", ""), fm["sub_points"])
                body = "\n\n".join([
                    "## 제시문\n\n" + b.get("제시문", "").strip(),
                    "## 논제\n\n" + prompt,
                    "## 모범답안\n\n" + b.get("모범답안", "").strip(),
                    "## 정답\n\n" + b.get("정답", "").strip(),
                    "## 채점표\n\n" + rubric,
                    "## 대표 오답\n\n" + b.get("대표 오답", d.get("wrong", "")).strip(),
                    "## 메모\n\n기존 ID " + lid + " (유형별 수리논술 N제, 2026-11-17판)에서 이관.",
                ]) + "\n"
                text_out = "---\n" + yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, width=1000) + "---\n\n" + body
                stats[sch] += 1
                scope_by_school[sch][scope] += 1
                books.append(iid)
                if not args.dry_run:
                    p = args.out / "items" / sch / tcode / f"{iid}.md"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(text_out, encoding="utf-8")
            vsrc = args.legacy_dir / folder / f"verify_{tcode}.py"
            if vsrc.exists() and not args.dry_run:
                (args.out / "verify").mkdir(exist_ok=True)
                shutil.copy(vsrc, args.out / "verify" / f"{sch}-{tcode}.py")
        if books and not args.dry_run:
            bk = {"code": f"{sch}-YUYU-v0", "title": f"{tax['schools'][sch]['name']} 유유상종 (유형별 N제) v0",
                  "series": "유유상종", "school": sch, "season": "2028", "edition": 0,
                  "items": [{"id": i, "chapter": i.split("-")[1], "ord": n + 1} for n, i in enumerate(books)]}
            (args.out / "books").mkdir(exist_ok=True)
            (args.out / "books" / f"{sch}-YUYU-v0.yaml").write_text(
                yaml.safe_dump(bk, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print("이관 문항 수:", dict(stats), "합계", sum(stats.values()))
    for sch, c in scope_by_school.items():
        print(f"  {sch} scope_2028 추정: {dict(c)}")
    for e in errors:
        print("ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
