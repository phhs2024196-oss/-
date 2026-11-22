#!/usr/bin/env python3
"""AJOU-G21M (아주대 2021학년도 논술 자연계열 저녁, 의학과 — 수학 문항 1) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

PREFIX = "AJOU-G21M"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


# ── 1번(의학과): 바둑돌 ──
def black_step(state, rule):
    a0, a1 = state
    if rule == 0 and a0 > 0:
        return (a0 - 1, a1 + 1)
    if rule == 1 and a1 > 0:
        return (a0 + 1, a1 - 1)
    if rule == 2 and a0 > 0:
        return (a0 - 1, a1)
    return state


def run_black(b, rules):
    s = (b, 0)
    for r in rules:
        s = black_step(s, r)
    return s


def given_log(fr: Fraction, table: dict) -> float:
    fn, fd = sp.factorint(fr.numerator), sp.factorint(fr.denominator)
    assert set(fn) | set(fd) <= set(table)
    return sum(e * table[p] for p, e in fn.items()) - sum(e * table[p] for p, e in fd.items())


def cond_prob_12():
    """사건 A(짝수 번째 직후 원점에 없음), B(12회 후 남아 있음) — 정확 DP."""
    dist = {(1, 0): Fraction(1)}
    for step in range(1, 13):
        nd = {}
        for s, pv in dist.items():
            for r in range(3):
                s2 = black_step(s, r)
                nd[s2] = nd.get(s2, 0) + pv / 3
        if step % 2 == 0:
            nd = {s: pv for s, pv in nd.items() if s[0] == 0}
        dist = nd
    pA = sum(dist.values()); pAB = sum(pv for s, pv in dist.items() if sum(s) > 0)
    return pAB / pA


def white_sum(nn, weight):
    tot = Fraction(0)
    for m in range(nn + 1):          # 홀수 눈 m 번 → 위치 nn-2m
        k = nn - 2 * m
        if k != 0:
            tot += weight(abs(k)) * Fraction(math.comb(nn, m), 2**nn)
    return tot


def configs(nb, nn):
    states = {((nb, 0), 0)}
    for _ in range(nn):
        nxt = set()
        for (bs, w) in states:
            for face in range(1, 7):
                nxt.add((black_step(bs, (face - 1) % 3), w + (1 if face % 2 == 0 else -1)))
        states = nxt
    return len(states)


def count_13():
    multi = len(set(itertools.permutations("aabb")))          # 한 규칙 자리(4칸)에 두 눈을 두 번씩
    total = 0
    for pos in itertools.combinations(range(12), 4):
        rest = [i for i in range(12) if i not in pos]
        for pos2 in itertools.combinations(rest, 4):
            rules = [2] * 12
            for i in pos:
                rules[i] = 0
            for i in pos2:
                rules[i] = 1
            s12 = run_black(4, rules)
            if sum(s12) < 4:
                continue
            for face in range(1, 7):      # 12회 후 흰 돌: 짝·홀 각 6번 → 원점
                if face % 2 == 0 and black_step(s12, (face - 1) % 3) == (0, 4):
                    total += multi ** 3
    return total


def p01():
    msgs, oks = [], []
    table = {2: 0.30, 3: 0.47, 7: 0.84, 11: 1.04}
    # 1-1 (1) 5회 후 남아 있지 않을 확률
    gone = sum(Fraction(1, 3**5) for rules in itertools.product(range(3), repeat=5) if run_black(1, rules) == (0, 0))
    lg = given_log(gone, table)
    oks.append(abs(lg - float(md_answer(1, "1-1"))) < 1e-9); msgs.append(f"(1)p={gone},log≈{lg:.2f}")
    # 1-1 (2) [증명-수치확인]
    pBA = cond_prob_12()
    mdexpr = sp.sympify(md_answer(1, "1-2").split("=", 1)[1].split(",")[0].replace("alpha", "(5/9)**5"))
    oks.append(pBA < Fraction(3, 92) and sp.Rational(pBA.numerator, pBA.denominator) == sp.nsimplify(mdexpr))
    msgs.append(f"(2)P(B|A)={float(pBA):.6f}<3/92 [증명-수치확인]")
    # 1-1 (3) 검은 돌 2개, 4회
    Xs = [sum(run_black(2, rules)) for rules in itertools.product(range(3), repeat=4)]
    E = Fraction(sum(Xs), 81); V = Fraction(sum(v * v for v in Xs), 81) - E**2
    d = kv(md_answer(1, "1-3"))
    oks.append(sp.Rational(E.numerator, E.denominator) == sp.sympify(d["E"]) and sp.simplify(sp.sqrt(sp.Rational(V.numerator, V.denominator)) - sp.sympify(d["sigma"])) == 0)
    msgs.append(f"(3)E={E},V={V}")
    # 1-2 (1) 홀수 n: Σ(k^2+1)(p_k+p_{-k})
    md = sp.sympify(md_answer(1, "2-1"), locals={"n": n})
    oks.append(all(white_sum(nn, lambda k: k * k + 1) == md.subs(n, nn) for nn in range(1, 20, 2))); msgs.append("(2-1)홀수 n=1..19 일치")
    # 1-2 (2)
    md = sp.sympify(md_answer(1, "2-2"), locals={"n": n})
    cs = {nn: configs(3, nn) for nn in range(3, 9)}
    oks.append(all(cs[nn] == md.subs(n, nn) for nn in cs)); msgs.append(f"(2-2)배치수{cs}")
    # 1-2 (3)
    c13 = count_13()
    oks.append(c13 == int(sp.sympify(md_answer(1, "2-3")))); msgs.append(f"(2-3)개수={c13}")
    report(1, all(oks), " ".join(msgs))


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
