#!/usr/bin/env python3
"""source_check 신뢰성 시험 — 정답이 알려진 오류를 DB 에 일부러 심고 몇 %를 잡는지 잰다.

사용: python3 tools/tests/test_source_check.py --pair G24A.pdf 'items/AJOU/PAST/AJOU-G24A-0*.yaml' [--pair …] [--n 40]
  * 기준선: 변조 전 실행에서 이미 '확인 필요'였던 필드는 어차피 사람이/모델이 보므로 '확인 범위 안'으로 따로 센다.
  * 핵심 지표 = 기준선에서 자동통과였던 필드에 심은 오류를 새로 잡은 비율(놓침률 = 1 − 이것).
변조 9종: 한글 글자 바꿈·한글 2~4자 삭제·숫자 바꿈·변수 바꿈·부호 뒤집기·부등호 뒤집기·
         문장 삭제·배점 바꿈·분수 위아래 바꿈
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import pathlib
import random
import re
import sys
import tempfile

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from source_check import Source, compare, item_fields, read_src  # noqa: E402
from textnorm import ANNOT_HEAD, plain  # noqa: E402

MATH = re.compile(r"\$[^$]+\$")


def in_math(text, fn):
    spans = [m.span() for m in MATH.finditer(text)]
    random.shuffle(spans)
    for a, b in spans:
        new = fn(text[a:b])
        if new and new != text[a:b]:
            return text[:a] + new + text[b:]
    return None


def in_annot(old, new):
    """바뀐 곳이 DB 전용 주석([그림 n: 설명], [판독확인: …]) 안이면 True — 원본 글 오류가 아니므로 시험에서 뺀다."""
    i = 0
    while i < min(len(old), len(new)) and old[i] == new[i]:
        i += 1
    j = 0
    while j < min(len(old), len(new)) - i and old[-1 - j] == new[-1 - j]:
        j += 1
    a, b = i, len(old) - j
    for m in ANNOT_HEAD.finditer(old):
        depth, k = 0, m.start()
        while k < len(old):
            depth += {"[": 1, "]": -1}.get(old[k], 0)
            k += 1
            if depth == 0:
                break
        if m.start() <= a and b <= k:
            return True
    return False


def mutate(kind, text):
    if kind == "hangul_sub":
        idx = [i for i, c in enumerate(text) if "가" <= c <= "힣"]
        if not idx:
            return None
        i = random.choice(idx)
        c = text[i]
        n = chr(0xAC00 + (ord(c) - 0xAC00 + random.randint(1, 11171)) % 11172)
        return text[:i] + n + text[i + 1:]
    if kind == "hangul_del":
        runs = [m for m in re.finditer(r"[가-힣]{4,}", text)]
        if not runs:
            return None
        m = random.choice(runs)
        k = random.randint(2, min(4, len(m.group()) - 1))
        s = random.randint(m.start(), m.end() - k)
        return text[:s] + text[s + k:]
    if kind == "digit":
        return in_math(text, lambda s: re.sub(r"\d", lambda m: str((int(m.group()) + random.randint(1, 9)) % 10), s, count=1)
                       if re.search(r"\d", s) else None)
    if kind == "var":
        def f(s):
            cands = [m for m in re.finditer(r"(?<![\\A-Za-z])[a-z](?![A-Za-z])", s)]
            if not cands:
                return None
            m = random.choice(cands)
            rep = random.choice([c for c in "abcknpqrstxyz" if c != m.group()])
            return s[:m.start()] + rep + s[m.end():]
        return in_math(text, f)
    if kind == "sign":
        def f(s):
            cands = [m for m in re.finditer(r"[+-]", s)]
            if not cands:
                return None
            m = random.choice(cands)
            return s[:m.start()] + ("-" if m.group() == "+" else "+") + s[m.end():]
        return in_math(text, f)
    if kind == "ineq":
        sw = {"<": ">", ">": "<", r"\le": r"\ge", r"\ge": r"\le", r"\leq": r"\geq", r"\geq": r"\leq"}
        def f(s):
            cands = [m for m in re.finditer(r"\\leq|\\geq|\\le(?![a-z])|\\ge(?![a-z])|<|>", s)]
            if not cands:
                return None
            m = random.choice(cands)
            return s[:m.start()] + sw[m.group()] + s[m.end():]
        return in_math(text, f)
    if kind == "sentence_del":
        parts = re.split(r"(?<=[.다])\s+", text.strip())
        if len(parts) < 2:
            return None
        i = random.randrange(len(parts))
        return " ".join(parts[:i] + parts[i + 1:])
    if kind == "frac_swap":
        def f(s):
            m = re.search(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", s)
            if not m or m.group(1) == m.group(2):
                return None
            return s[:m.start()] + m.group(0).replace("{" + m.group(1) + "}{" + m.group(2) + "}",
                                                       "{" + m.group(2) + "}{" + m.group(1) + "}") + s[m.end():]
        return in_math(text, f)
    return None


KINDS = ["hangul_sub", "hangul_del", "digit", "var", "sign", "ineq", "sentence_del", "points", "frac_swap"]


def text_slots(it):
    for i, s in enumerate(it.get("stimuli") or []):
        yield f"stimuli[{s.get('label')}]", ("stimuli", i, "body")
    yield "stem", ("stem",)
    for i, s in enumerate(it.get("subparts") or []):
        yield f"sub[{s['id']}]", ("subparts", i, "stem")


def get(it, path):
    o = it
    for k in path:
        o = o[k]
    return o


def setv(it, path, v):
    o = it
    for k in path[:-1]:
        o = o[k]
    o[path[-1]] = v


def flagged(res, lenient=False):
    return {n for g, n, _ in res if not (lenient and g == "ORDER")}


def run(src_raw, paths):
    src = Source(src_raw, "pdf")
    return compare(src, [f for p in paths for f in item_fields(pathlib.Path(p))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True)
    ap.add_argument("--n", type=int, default=40, help="시험지당 변조 횟수")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    stat = collections.defaultdict(lambda: [0, 0, 0])   # kind → [자동통과 필드에 심음, 잡음, 이미 확인범위]
    base_tot = base_ok = 0
    for srcp, pat in a.pair:
        raw = read_src(srcp)
        paths = sorted(glob.glob(pat))
        items = {p: yaml.safe_load(pathlib.Path(p).read_text(encoding="utf-8")) for p in paths}
        base = run(raw, paths)
        bflag = flagged(base)
        names = [f"{it['id']} {lbl}" for it in items.values() for lbl, _ in text_slots(it)]
        base_tot += len(names)
        base_ok += sum(n not in bflag for n in names)
        with tempfile.TemporaryDirectory() as td:
            for _ in range(a.n):
                p = random.choice(paths)
                it = copy.deepcopy(items[p])
                kind = random.choice(KINDS)
                if kind == "points":
                    subs = it.get("subparts") or []
                    if not subs:
                        continue
                    s = random.choice(subs)
                    s["points"] = int(s["points"]) + random.choice([-3, -2, -1, 1, 2, 3])
                    target = f"{it['id']} sub[{s['id']}]"
                else:
                    slots = list(text_slots(it))
                    random.shuffle(slots)
                    for lbl, path in slots:
                        old = get(it, path) or ""
                        new = mutate(kind, old)
                        # 그림 설명·판독 주석 안만 바뀐 변조는 원본 글 오류가 아니므로 제외
                        if new and plain(new) != plain(old) and not in_annot(old, new):
                            setv(it, path, new)
                            target = f"{it['id']} {lbl}"
                            break
                    else:
                        continue
                mp = pathlib.Path(td) / pathlib.Path(p).name
                mp.write_text(yaml.dump(it, allow_unicode=True, sort_keys=False), encoding="utf-8")
                res = run(raw, [str(mp) if q == p else q for q in paths])
                newmsgs = {(g, n, m) for g, n, m in res} - {(g, n, m) for g, n, m in base}
                caught = bool(newmsgs)
                if target in bflag:
                    stat[kind][2] += 1
                else:
                    stat[kind][0] += 1
                    stat[kind][1] += caught
                    if not caught:
                        print(f"놓침 {kind} {target}")
    print(f"기준선: 필드 {base_tot}개 중 자동통과 {base_ok} ({base_ok / max(1, base_tot):.0%})")
    T = C = 0
    for k in KINDS:
        t, c, already = stat[k]
        T += t
        C += c
        print(f"  {k:13} 자동통과 필드에 심음 {t:3} · 잡음 {c:3} ({c / t:.0%})" if t else f"  {k:13} 자동통과 필드 대상 없음",
              f"· 이미 확인범위 {already}")
    print(f"SUMMARY 검출률 {C}/{T} = {C / max(1, T):.1%} (자동통과 필드 기준, 놓침률 {1 - C / max(1, T):.1%})")


if __name__ == "__main__":
    main()
