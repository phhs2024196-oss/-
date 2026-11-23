#!/usr/bin/env python3
"""KUT-M23A-01~06 (한국기술교육대 2023학년도 수시 모의 논술고사 【수리】모의문제, 6월 배포) — sympy 검산. 기대값은 YAML 에서 V.ans() 로 읽는다."""
import re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M23A", "items/KUT/PAST")
a, tt = sp.symbols("a t", real=True)


def p01():
    f = x**3 - 3 * a * x**2 + 2 * a**2 * x - 2 * a**3
    slope = sp.diff(f, x).subs(x, tt)
    tmin = sp.solve(sp.diff(slope, tt), tt)[0]          # 기울기 최소가 되는 t
    A = (tmin, f.subs(x, tmin))
    m = slope.subs(tt, tmin)
    xint = sp.solve(m * (x - A[0]) + A[1], x)[0]       # 접선의 x절편 g(a)
    V.check(1, "g'(2)", sp.diff(xint, a).subs(a, 2), V.ans(1, "1"))


def p02():
    D = sp.expand((a - 2)**2 - 16)
    rng = sp.solve_univariate_inequality(D < 0, a, relational=False)
    lo, hi = [sp.Integer(v) for v in re.findall(r"-?\d+", V.ans(2, "1"))]
    V.report(2, rng == sp.Interval.open(lo, hi), f"D<0 해집합 {rng} / YAML {V.ans(2, '1')}")


def p03():
    y = -sp.cos(x)**2 - 3 * sp.sin(x) + 8
    lo = sp.minimum(y, x, sp.Interval(0, 2 * sp.pi))
    V.check(3, "k max", lo, V.ans(3, "1"))


def p04():
    A, d, r = sp.symbols("A d r")
    sols = sp.solve([A - 2 * r, A + d - 2, A + 2 * d - 2 * r**2], [A, d, r], dict=True)
    sols = [s for s in sols if s[r] != 1]
    s = sols[0]
    a10 = s[A] + 9 * s[d]
    b10 = 2 * s[r]**9
    V.check(4, "|a10|+|b10|", abs(a10) + abs(b10), V.ans(4, "1"))


def p05():
    A, B = sp.symbols("A B")
    f = x**3 + 2 * A * x
    g = 3 * x**2 + B * x
    sol = sp.solve([sp.integrate(g, (x, 0, 1)) - A, sp.integrate(f, (x, 0, 1)) - B], [A, B], dict=True)[0]
    eq = sp.expand((f - x * g).subs(sol))
    roots = sp.solve(eq, x)
    V.check(5, "실근의 합", sum(roots), V.ans(5, "1"))


def p06():
    from mpmath import mp, mpf, log, nint
    mp.dps = 60
    ns = []
    for n in range(3, 20000):
        v = log(3 * n * log(n) / log(3)) / log(9)
        if abs(v - nint(v)) < mpf(10)**-40 and nint(v) >= 1:
            ns.append(n)
    assert len(ns) >= 2
    V.check(6, "81 a1/a2", sp.Rational(81 * ns[0], ns[1]), V.ans(6, "1"))


if __name__ == "__main__":
    V.run(globals())
