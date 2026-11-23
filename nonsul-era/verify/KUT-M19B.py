#!/usr/bin/env python3
"""KUT-M19B (2019 논술 예시문항 수학-B형 01~03) 검산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy import Rational as R, symbols, solve, sqrt, summation, oo, simplify, Piecewise, integrate
def same_set(cond, text, a):
    return sp.solve_univariate_inequality(S(text.replace('a', 'a')) , a) == cond

V = Verifier("KUT-M19B", "items/KUT/PAST")

def p01():
    xx, k = symbols("x k", real=True)
    nn = symbols("n", positive=True, integer=True)
    r2 = (R(4, 5)) ** nn
    disc = sp.discriminant(xx**2 + (-2 * xx + k) ** 2 - r2, xx)
    ks = solve(disc, k)
    kpos = [s for s in ks if s.subs(nn, 1) > 0][0]
    V.check(1, "1-1 a_n (n=1..6)", [simplify(kpos.subs(nn, m)) for m in range(1, 7)], [S(V.ans(1, "1-1")).subs(sp.Symbol("n"), m) for m in range(1, 7)])
    a = lambda m: sqrt(5) * (2 / sqrt(5)) ** m
    # 수치 합(부분합 -> 극한)
    ssum = sum(sp.N((a(m)) ** 2 * (sqrt(5) / 2) ** (m - 1), 30) for m in range(1, 400))
    V.check(1, "1-2 합(수치)", sp.N(ssum, 20), sp.N(S(V.ans(1, "1-2")), 20), tol=1e-8)
    V.check(1, "1-2 기호", 4 / (1 - 2 / sqrt(5)), V.ans(1, "1-2"))

def p02():
    xx, a = symbols("x a", real=True)
    f = xx**3 - 6 * xx**2 + a * xx - 1
    disc = sp.discriminant(f.diff(xx), xx)
    cond = sp.solve_univariate_inequality(disc > 0, a)
    V.report(2, sp.simplify(sp.Eq(sp.Symbol("a") < 12, S(V.ans(2, "1-1")))) is sp.true or same_set(cond, V.ans(2, "1-1"), a), f"1-1 범위 {cond} / YAML {V.ans(2,'1-1')}")
    f9 = f.subs(a, 9)
    mn, mx = extrema(f9, xx, 0, 4)
    V.check(2, "1-2 max", mx, V.ans(2, "1-2", "max")); V.check(2, "1-2 min", mn, V.ans(2, "1-2", "min"))
    # 수치: a=11.9 극값 2개, a=12 극값 없음
    cnt = lambda av: len(set(sp.real_roots(f.diff(xx).subs(a, av))))
    V.report(2, cnt(R(119, 10)) == 2 and cnt(12) == 1, "a<12 경계 확인")

def p03():
    xx, a = symbols("x a", real=True)
    f1 = a / 5 * (xx + 5); f2 = -a / 15 * (xx - 15)
    aa = solve(integrate(f1, (xx, -5, 0)) + integrate(f2, (xx, 0, 15)) - 1, a)[0]
    V.check(3, "a", aa, R(1, 10))
    V.check(3, "1-1 P(X<=0)", integrate(f1.subs(a, aa), (xx, -5, 0)), V.ans(3, "1-1"))
    V.check(3, "1-2 P(X>=10)", integrate(f2.subs(a, aa), (xx, 10, 15)), V.ans(3, "1-2"))

if __name__ == "__main__":
    V.run(globals())
