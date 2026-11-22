#!/usr/bin/env python3
"""EWHA-M22 (이화여대 2022학년도 수시 모의논술 자연계열) — sympy 검산.
규칙: 정답 하드코딩 비교 금지. 문항 YAML 의 answer 를 읽어 설정으로부터 독립 재계산한 값과 비교.
증명 소문항은 수치 반례 탐색·작은 지수 일반화 [증명-수치확인]."""
import math
import pathlib
import random
import re
import sys

import mpmath as mp
import sympy as sp
import yaml

PREFIX = "EWHA-M22"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    rng = random.Random(7)
    # (1) e^x - e^a >= e^a (x-a) : 격자·무작위 [증명-수치확인]
    ok1 = all(math.exp(x) - math.exp(a) >= math.exp(a) * (x - a) - 1e-9 * max(1, math.exp(max(x, a)))
              for a in [i / 10 for i in range(-50, 51)] for x in [j / 10 for j in range(-50, 51)])
    # (2) 접선과 y=1 교점 b 를 sympy 로 구해 e^b>=1 확인
    a, x = sp.symbols("a x", real=True)
    tangent = sp.exp(a) * (x - a) + sp.exp(a)
    b = sp.solve(sp.Eq(tangent, 1), x)[0]
    ok2 = all(sp.exp(b.subs(a, v)).evalf() >= 1 - 1e-12 for v in [rng.uniform(-30, 30) for _ in range(500)] + [0])
    # (3) x_1=2022, 300항: x_n>=0, 감소 (mpmath 고정밀)
    mp.mp.dps = 50
    bl = sp.lambdify(a, b, "mpmath")
    xn = mp.mpf(2022)
    ok3 = True
    for _ in range(3000):
        nx = bl(xn)
        ok3 &= nx >= 0 and nx <= xn
        xn = nx
    report(1, ok1 and ok2 and ok3, f"b={b} [증명-수치확인] x_3000~{mp.nstr(xn, 5)}")


def _beta(m, n):
    return sp.factorial(m) * sp.factorial(n) / sp.factorial(m + n + 1)


def p02():
    # 정확값(베타 적분 공식)으로 지수 2021~2023 관계 확인
    A, B, C, D = _beta(2021, 2021), _beta(2022, 2022), _beta(2022, 2021), _beta(2023, 2021)
    loc = {"A": A, "B": B, "C": C, "D": D}

    def holds(expr):
        lhs, rhs = expr.split("=")
        return sp.simplify(sp.sympify(lhs, locals=loc) - sp.sympify(rhs, locals=loc)) == 0
    ok = holds(md_answer(2, "1")) and holds(md_answer(2, "2")) and \
        all(holds(part) for part in md_answer(2, "3").split(";"))
    # 작은 지수 n 에 대해 같은 구조를 sympy integrate 로 직접 확인
    x = sp.symbols("x")
    for n in range(1, 7):
        I = lambda p, q: sp.integrate(x**p * (1 - x)**q, (x, 0, 1))
        a_, b_, c_, d_ = I(n, n), I(n + 1, n + 1), I(n + 1, n), I(n + 2, n)
        ok &= b_ + d_ == c_ and b_ == sp.Rational(n + 1, n + 2) * d_ and a_ - b_ - c_ == d_
        ok &= b_ / a_ == sp.Rational(n + 1, 2 * (2 * n + 3))   # n=2021 → 1011/4045
    report(2, ok, f"B/A={sp.nsimplify(B / A)} (베타 정확값 + n=1..6 직접 적분)")


def _segdist2(a):
    """(a,0)에서 선분 (t,t+2), -1<=t<=1 까지 최단거리 제곱: 3분 탐색 + 끝점."""
    g = lambda t: (t - a) ** 2 + (t + 2) ** 2
    lo, hi = -1.0, 1.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if g(m1) < g(m2):
            hi = m2
        else:
            lo = m1
    return min(g(lo), g(-1.0), g(1.0))


def p03():
    a, t = sp.symbols("a t", real=True)
    foot_t = sp.solve(sp.Eq((t + 2 - 0) / (t - a), -1), t)[0]
    md1 = sp.sympify(md_answer(3, "1"), locals={"a": a})
    ok1 = all(sp.simplify(u - w) == 0 for u, w in zip(md1, (foot_t, foot_t + 2)))
    f2 = sp.sympify(md_answer(3, "2"), locals={"a": a})
    f3 = sp.sympify(md_answer(3, "3"), locals={"a": a})
    ok2 = all(abs(_segdist2(v) - float(f2.subs(a, v))) < 1e-9 for v in (-7, -3.2, -1, -0.4, -1e-3))
    ok3 = all(abs(_segdist2(v) - float(f3.subs(a, v))) < 1e-9 for v in (0, 0.7, 2, 3.9, 4, 4.01, 6.5, 20))
    report(3, ok1 and ok2 and ok3, f"foot t={foot_t} f(-3.2)~{_segdist2(-3.2):.6f} f(6.5)~{_segdist2(6.5):.6f}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
