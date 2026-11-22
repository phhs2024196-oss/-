#!/usr/bin/env python3
"""dup_check 신뢰성 시험 — DB 문항으로 '숨긴 변형'을 만들어 원 문항을 FLAG 로 잡는지 잰다.

사용: python3 tools/tests/test_dup_check.py --db <폴더> [--db …] [--n 60]
변형(쉬움→어려움): 숫자 바꿈 · 변수 이름 바꿈 · 문장 30% 삭제+순서 섞기 · 소문항 하나만 떼어 씀 ·
                  셋 다(숫자+변수+문장) · 셋 다+본문 절반만
오경보: 실제 문항끼리(자기 제외) FLAG 비율 — 같은 학교의 공통 제시문 등 '진짜로 겹치는' 쌍도 섞여 있으니 표본을 출력해 본다.
"""
from __future__ import annotations

import argparse
import collections
import copy
import pathlib
import random
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dup_check import Doc, check, load  # noqa: E402

MATH = re.compile(r"\$[^$]+\$")


def map_math(text, fn):
    return MATH.sub(lambda m: fn(m.group(0)), text)


def renumber(it):
    k = random.randint(1, 9)
    f = lambda s: re.sub(r"\d", lambda m: str((int(m.group()) + k) % 10), s)  # noqa: E731
    return apply(it, lambda t: map_math(t, f))


def rename(it):
    letters = list("abcdefghkmnpqrstuvwxyz")
    perm = letters[:]
    random.shuffle(perm)
    tbl = dict(zip(letters, perm))
    f = lambda s: re.sub(r"(?<![\\A-Za-z])[a-z](?![A-Za-z])", lambda m: tbl.get(m.group(), m.group()), s)  # noqa: E731
    return apply(it, lambda t: map_math(t, f))


def paraphrase(it, keep=0.7):
    def f(t):
        sents = re.split(r"(?<=[.다])\s+", t)
        sents = [s for s in sents if random.random() < keep] or sents[:1]
        random.shuffle(sents)
        return " ".join(sents)
    return apply(it, f)


def halve(it):
    return apply(it, lambda t: t[: max(1, len(t) // 2)])


def one_sub(it):
    it = copy.deepcopy(it)
    subs = it.get("subparts") or []
    if len(subs) > 1:
        it["subparts"] = [random.choice(subs)]
    it["stem"] = "다음 물음에 답하시오."
    return it


def apply(it, fn):
    it = copy.deepcopy(it)
    for s in it.get("stimuli") or []:
        s["body"] = fn(s.get("body") or "")
    it["stem"] = fn(it.get("stem") or "")
    for s in it.get("subparts") or []:
        s["stem"] = fn(s.get("stem") or "")
    return it


VARIANTS = {
    "숫자": renumber,
    "변수": rename,
    "문장30%삭제+섞기": paraphrase,
    "소문항하나만": one_sub,
    "숫자+변수+문장": lambda it: paraphrase(rename(renumber(it))),
    "숫자+변수+문장+절반": lambda it: halve(paraphrase(rename(renumber(it)))),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="append", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    random.seed(a.seed)
    db = load(a.db)
    raw = {d.id: yaml.safe_load(d.path.read_text(encoding="utf-8")) for d in db}
    print(f"DB {len(db)}문항")
    for name, fn in VARIANTS.items():
        hit = top1 = 0
        for d in random.sample(db, min(a.n, len(db))):
            v = fn(raw[d.id])
            v["id"] = "QUERY"
            q = Doc(pathlib.Path("q.yaml"), v)
            flagged, top = check(q, db, 5)
            hit += any(r[1].id == d.id for r in flagged)
            top1 += bool(top) and top[0][1].id == d.id
        n = min(a.n, len(db))
        print(f"  {name:18} 원문항 FLAG {hit}/{n} ({hit / n:.0%}) · 1순위 {top1}/{n}")
    # 오경보: 실제 문항끼리
    fp = 0
    samp = random.sample(db, min(a.n, len(db)))
    pairs = collections.Counter()
    ex = []
    for d in samp:
        flagged, _ = check(d, db, 5)
        if flagged:
            fp += 1
            same_school = all(r[1].school == d.school for r in flagged)
            pairs["같은 학교" if same_school else "다른 학교 포함"] += 1
            if len(ex) < 6:
                r = flagged[0]
                ex.append(f"    {d.id} ↔ {r[1].id} ({'·'.join(r[2])}) 글 {r[3]['글']:.2f} 골격 {r[3]['골격']:.2f}")
    print(f"  실제 문항 FLAG {fp}/{len(samp)} ({fp / len(samp):.0%}) {dict(pairs)} — 표본:")
    print("\n".join(ex))


if __name__ == "__main__":
    main()
