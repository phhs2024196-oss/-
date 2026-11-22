#!/usr/bin/env python3
"""AJOU-M22 (아주대 2022 모의 자연) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-M22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(no: int, sub: str) -> dict:
    """'a=..; b=..' 꼴 answer.value 를 사전으로."""
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

import itertools
import math
import random
from fractions import Fraction


def _b_seq(a):
    """제시문의 정의대로 제1유도수열."""
    b = [Fraction(a[0])]
    for ak in a[1:]:
        b.append(ak - 1 / b[-1])
    return b


def p01():
    kk = sp.symbols("k", positive=True, integer=True)
    # (1-1) [증명-수치확인] n=4..60: 분자(기약분수) 끝항 짝수, 나머지 홀수
    ok1 = True
    for N in range(4, 61):
        c = [q.numerator for q in _b_seq([3] + [2] * (N - 2) + [3])]
        ok1 = ok1 and c[-1] % 2 == 0 and all(v % 2 == 1 for v in c[:-1])
    # (1-2) b_k 식 (k=1..80)
    mdb = sp.sympify(md_answer(1, "2"), locals={"k": kk})
    bs = _b_seq([2] * 80)
    ok2 = all(sp.Rational(bs[i].numerator, bs[i].denominator) == mdb.subs(kk, i + 1) for i in range(80))
    # (1-3) 끝항 25/4: 항 2..9, 길이 1..6 전수 탐색 → 해가 유일하고 md 와 같음
    sols = [a for L in range(1, 7) for a in itertools.product(range(2, 10), repeat=L) if _b_seq(a)[-1] == Fraction(25, 4)]
    md3 = tuple(int(v) for v in sp.sympify(md_answer(1, "3")))
    ok3 = sols == [md3]
    # (1-4) [증명-수치확인] 역추적 절차(d=ceil)로 s/t (2<=s<=80, t<s, 서로소) 모두 기본수열 구성
    ok4 = True
    for s_ in range(2, 81):
        for t_ in range(1, s_):
            if math.gcd(s_, t_) != 1:
                continue
            q, seq = Fraction(s_, t_), []
            while q.denominator != 1:
                d = -(-q.numerator // q.denominator)
                seq.append(d)
                q = 1 / (d - q)
            seq.append(int(q))
            a = seq[::-1]
            ok4 = ok4 and all(v > 1 for v in a) and _b_seq(a)[-1] == Fraction(s_, t_)
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] (1-1) n=4..60, (1-3) 해={sols}, (1-4) s<=80 전부 구성")


def p02():
    random.seed(1)
    ok = True
    # (2-1)(1)~(3), (2-2) [증명-수치확인] 실근만 갖는 무작위 다항식
    for _ in range(40):
        deg = random.randint(2, 6)
        roots = [random.choice([-3, -2, -1, 0, 1, 2, 3, sp.Rational(1, 2)]) for _ in range(deg)]
        f = sp.expand(sp.prod([x - r for r in roots]))
        g = sp.expand(sp.diff(f, x) - f)
        gr = len(sp.real_roots(sp.Poly(g, x)))                            # 중복 포함 실근 수
        ok = ok and gr == deg
        distinct = sorted(set(roots))
        for r in distinct:
            m = roots.count(r)
            if m > 1:
                mult_g = sp.roots(sp.Poly(g, x)).get(r, 0)
                ok = ok and mult_g == m - 1                               # (1)
        for u, v in zip(distinct, distinct[1:]):
            if roots.count(u) == 1:                                       # (2)(3): u 단근
                ok = ok and sp.Poly(g, x).count_roots(u, v) - (1 if g.subs(x, v) == 0 else 0) >= 1
        a_, b_ = random.choice([-2, -1, 1, 2, 3]), random.choice([-3, -1, 1, 2])
        h = sp.expand(sp.diff(f, x, 2) - (a_ + b_) * sp.diff(f, x) + a_ * b_ * f)
        ok = ok and len(sp.real_roots(sp.Poly(h, x))) == deg              # (2-2) 중복 포함
    # (2-3) md 의 예가 조건을 만족하는지
    ex = kv(2, "3")
    c, d = sp.sympify(ex["c"]), sp.sympify(ex["d"])
    f = sp.sympify(ex["f"], locals={"x": x})
    h = sp.expand(sp.diff(f, x, 2) - (c + d) * sp.diff(f, x) + c * d * f)
    ok3 = c != 0 and d != 0 and sp.degree(f, x) == 2 and sp.Poly(f, x).count_roots() == 0 \
        and len(sp.real_roots(sp.Poly(h, x))) == 2
    report(2, ok and ok3, f"[증명-수치확인] 무작위 40개 다항식, (2-3) h={sp.factor(h)}")


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
