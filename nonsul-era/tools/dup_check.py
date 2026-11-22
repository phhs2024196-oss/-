#!/usr/bin/env python3
"""문제집용 중복·유사 문항 검사기 — 새 문항이 DB(기출·자체 문항)와 겹치는지. 놓침 최소화(OR 규칙).

사용:
  python3 tools/dup_check.py NEW.yaml [NEW2.yaml …] [--db items] [--db /경로/다른DB …] [--top 5]
  python3 tools/dup_check.py --all [--db items]          # DB 안에서 서로 겹치는 쌍 전부(세트 점검용)
판정 신호(하나라도 걸리면 FLAG — 놓치지 않는 쪽으로):
  (공통) DB 문항 1%(최소 4문항) 넘게 나오는 조각은 상투구로 보고 뺀다
  글     한글 3글자 조각 포함률(짧은 쪽 기준) ≥ 0.25, 겹친 조각 3개 이상 — 같은 지문·발문 재사용
  골격   수식 골격(숫자→#, 변수명→등장 순서 번호) 4-조각 포함률 ≥ 0.35, 3개 이상 — 숫자·문자만 바꾼 변형
  식     8자 이상, 상투구 아닌 같은 수식이 2개 이상              — 핵심 식 재사용
  정보부족 질의의 비상투구 조각이 너무 적으면(8개 미만) 자동 판정 불가 → 상위 후보를 FLAG 로 내서 직접 확인
  답+발상 같은 답(3자 이상)이 있고 발상·소재 태그가 겹침(자카드 ≥ 0.3)
출력: 질의 문항마다 FLAG/OK + 상위 후보(점수·걸린 신호). 신뢰성 시험: tools/tests/test_dup_check.py
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from textnorm import hangul, math_seq, plain  # noqa: E402

KIT = pathlib.Path(__file__).resolve().parent.parent
# 문턱값: tools/tests/test_dup_check.py 로 482문항에서 맞춤(2026-11-22) — 변형 360건 중 356건 FLAG, 원 문항은 전부 1순위
TH = {"글": 0.25, "골격": 0.35, "식": 2, "태그": 0.3, "글최소": 3, "골격최소": 3, "정보부족": 8}
COMMON = 0.01      # DB 문항의 1% 넘게(최소 4문항) 나오는 조각은 상투구 → 점수에서 뺀다
MATH = re.compile(r"\$([^$]+)\$")


def texts(it: dict) -> list[str]:
    out = [s.get("body") or "" for s in it.get("stimuli") or []] + [it.get("stem") or ""]
    out += [s.get("stem") or "" for s in it.get("subparts") or []]
    return out


def skeleton(seq: list[str]) -> list[str]:
    """숫자 → #, 한 글자 변수 → v1,v2…(문항 안 첫 등장 순서). 함수명(sin 등)은 글자 단위라 그대로 섞이지만
    양쪽에 같은 규칙이 적용되므로 비교에는 문제없다."""
    names, out = {}, []
    for t in seq:
        if t.isdigit():
            if not out or out[-1] != "#":
                out.append("#")
        elif t.isascii() and t.isalpha():
            names.setdefault(t, f"v{len(names)}")
            out.append(names[t])
        else:
            out.append(t)
    return out


def shingles(seq, k: int) -> set:
    return {tuple(seq[i:i + k]) for i in range(max(0, len(seq) - k + 1))}


class Doc:
    def __init__(self, path: pathlib.Path, it: dict):
        self.path, self.id = path, it.get("id", path.stem)
        raw = "\n".join(texts(it))
        p = plain(raw, "latex")
        h = hangul(p)
        self.h3 = shingles(h, 3)
        self.sk = shingles(skeleton(math_seq(p)), 4)
        self.ex = {re.sub(r"\s+", "", m) for m in MATH.findall(raw) if len(re.sub(r"\s+", "", m)) >= 8}
        self.ans = {re.sub(r"\s+", "", str((s.get("answer") or {}).get("value") or ""))
                    for s in it.get("subparts") or []}
        self.ans = {a for a in self.ans if len(a) >= 3 and a.lower() not in {"null", "none", "todo"}}
        el = it.get("elements") or {}
        self.tags = {str(t) for t in (it.get("idea_tags") or [])} | \
                    {str(t) for t in (el.get("materials") or []) + (el.get("concepts") or [])}
        self.school = it.get("school")


def contain(a: set, b: set) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def jacc(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class Common:
    """상투구 목록(문서 빈도). '다음 물음에 답하시오', 'f(x)=' 같은 조각이 점수를 부풀리지 않게."""
    sets: dict = {"h3": set(), "sk": set(), "ex": set()}

    @classmethod
    def build(cls, db: list) -> None:
        import collections
        lim = max(4, int(len(db) * COMMON))
        for k in cls.sets:
            c = collections.Counter(x for d in db for x in getattr(d, k))
            cls.sets[k] = {x for x, v in c.items() if v > lim}


def rare(d, k: str) -> set:
    return getattr(d, k) - Common.sets[k]


def score(q: Doc, d: Doc) -> tuple[float, list[str], dict]:
    qh, dh, qs, ds = rare(q, "h3"), rare(d, "h3"), rare(q, "sk"), rare(d, "sk")
    s = {"글": contain(qh, dh) if len(qh & dh) >= TH["글최소"] else 0.0,
         "골격": contain(qs, ds) if len(qs & ds) >= TH["골격최소"] else 0.0,
         "정보부족": len(qh) < TH["정보부족"] and len(qs) < TH["정보부족"],
         "식": len(rare(q, "ex") & rare(d, "ex")),
         "답": len(q.ans & d.ans), "태그": jacc(q.tags, d.tags)}
    hits = []
    if s["글"] >= TH["글"]:
        hits.append("글")
    if s["골격"] >= TH["골격"]:
        hits.append("골격")
    if s["식"] >= TH["식"]:
        hits.append("식")
    if s["답"] and s["태그"] >= TH["태그"]:
        hits.append("답+발상")
    total = max(s["글"], s["골격"]) + 0.1 * min(s["식"], 5) + 0.2 * bool(s["답"]) + 0.2 * s["태그"]
    return total, hits, s


def load(paths: list[str]) -> list[Doc]:
    docs = []
    for root in paths:
        r = pathlib.Path(root)
        files = [r] if r.is_file() else sorted(r.rglob("*.yaml"))
        for f in files:
            try:
                it = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(it, dict) and it.get("subparts"):
                docs.append(Doc(f, it))
    return docs


def check(q: Doc, db: list[Doc], top: int = 5):
    if not any(Common.sets.values()):
        Common.build(db)
    rows = []
    for d in db:
        if d.id == q.id:
            continue
        tot, hits, s = score(q, d)
        rows.append((tot, d, hits, s))
    rows.sort(key=lambda r: -r[0])
    flagged = [r for r in rows if r[2]]
    if rows and rows[0][3]["정보부족"] and not flagged:
        flagged = [(r[0], r[1], ["정보부족(직접 확인)"], r[3]) for r in rows[:top]]
    return flagged, rows[:top]


def fmt(r) -> str:
    tot, d, hits, s = r
    return (f"{d.id:16} 글 {s['글']:.2f} 골격 {s['골격']:.2f} 식 {s['식']} 답 {s['답']} 태그 {s['태그']:.2f}"
            f"  {'⚑ ' + '·'.join(hits) if hits else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", nargs="*")
    ap.add_argument("--db", action="append", help="비교 대상 폴더/파일(여러 번). 기본 items/")
    ap.add_argument("--all", action="store_true", help="DB 안의 모든 쌍")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    db = load(a.db or [str(KIT / "items")])
    queries = db if a.all else load(a.new)
    nflag, seen = 0, set()
    for q in queries:
        flagged, top = check(q, db, a.top)
        if a.all:
            for r in flagged:
                key = tuple(sorted((q.id, r[1].id)))
                if key not in seen:
                    seen.add(key)
                    print(f"FLAG {q.id} ↔ {fmt(r)}")
            nflag += bool(flagged)
            continue
        print(f"{'FLAG' if flagged else 'OK  '} {q.id}")
        for r in (flagged[:a.top] if flagged else top[:3]):
            print("   ", fmt(r))
        nflag += bool(flagged)
    print(f"SUMMARY dup_check: 질의 {len(queries)} · FLAG {nflag}" + (f" · 쌍 {len(seen)}" if a.all else "")
          + f" · DB {len(db)}문항")
    return 1 if nflag else 0


if __name__ == "__main__":
    sys.exit(main())
