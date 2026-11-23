#!/usr/bin/env python3
"""KUT-M25 (2025학년도 한국기술교육대 모의논술 01~10) sympy 검산 — 기대값은 YAML 에서 읽는다."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M25", "items/KUT/PAST")


def p01():
    a = sp.symbols("a", positive=True)
    f = lambda z: a**z
    a0 = [s for s in sp.solve(sp.Eq(f(f(0)), sp.sqrt(5)), a) if s > 1][0]
    V.check(1, "f(3)", (a0**3), V.ans(1, "1"))


def p02():
    xs = [i for i in range(-50, 51) if i > 0 and i != 1 and -i**2 + 5*i + 14 > 0]
    V.check(2, "정수 x 합", sum(xs), V.ans(2, "1"))


def p03():
    c = sp.symbols("c")
    g = -c**2 + c
    M, m = extrema(g, c, -1, 1)[1], extrema(g, c, -1, 1)[0]
    V.check(3, "M+m", M + m, V.ans(3, "1"))
    import numpy as np
    xs = np.linspace(0, 2*np.pi, 200001); ys = -np.cos(xs)**2 + np.cos(xs)
    V.check(3, "수치 M+m", sp.nsimplify(round(ys.max()+ys.min(), 6)), V.ans(3, "1"))


def p04():
    a, b = sp.symbols("a b")
    f = 2*x**2 + a*x + b
    # x->oo: (f-2x^2)/(x-1) -> a
    sol = sp.solve([sp.limit((f - 2*x**2)/(x - 1), x, sp.oo) - 4, sp.limit((x + 2)/f, x, 1) - sp.Rational(3, 5)], [a, b], dict=True)[0]
    V.check(4, "f(2)", f.subs(sol).subs(x, 2), V.ans(4, "1"))


def p05():
    f = x**3 - 3*x**2 + 6
    al, be = sorted(sp.solve(sp.diff(f, x), x))
    V.check(5, "기울기", (f.subs(x, be) - f.subs(x, al))/(be - al), V.ans(5, "1"))


def p06():
    a, b = sp.symbols("a b", real=True)
    f0 = a - b
    g0 = f0**2 + b*f0
    L = sp.limit((-5*x - 6)**2 + b*(-5*x - 6), x, 0)
    R = sp.limit((x + 2)**2 + b*(x + 2), x, 0)
    sols = sp.solve([sp.Eq(g0, L), sp.Eq(L, R)], [a, b], dict=True)
    sols = [s for s in sols if s[a] > 0]
    V.check(6, "a+b", (sols[0][a] + sols[0][b]), V.ans(6, "1"))


def p07():
    a, b, c = sp.symbols("a b c")
    fp = a*x**2 + b*x + c    # f' = 3A x^2+2B x+C 대신 f'를 직접 미지 이차식으로
    s = sp.solve([fp.subs(x, -3), fp.subs(x, -1), fp.subs(x, 0) + 3], [a, b, c], dict=True)[0]
    f = sp.integrate(fp.subs(s), x)           # f(0)=0
    crit = sp.solve(sp.diff(f, x), x)
    ks = sorted(f.subs(x, r) for r in crit)
    # 서로 다른 두 실근 <=> k 가 극값
    V.check(7, "k 합", sum(ks), V.ans(7, "1"))


def p08():
    t = sp.symbols("t", real=True, nonnegative=True)
    v1, v2 = t**2 + 2*t, 3*t**2 + t
    ts = [r for r in sp.solve(sp.Eq(v1, v2), t) if r > 0][0]
    d = abs(sp.integrate(v1, (t, 0, ts)) - sp.integrate(v2, (t, 0, ts)))
    V.check(8, "거리", d, V.ans(8, "1"))


def p09():
    # a_n 을 점화 (3n+5)a_{n+1}=(3n-1)a_n 로 생성하고 원식을 수치 확인
    a = {1: sp.Integer(4)}
    for n_ in range(1, 12):
        a[n_ + 1] = a[n_]*(3*n_ - 1)/(3*n_ + 5)
    V.check(9, "1-1 a1", sp.Integer(20)/5, V.ans(9, "1-1"))
    S = lambda n_: sum(a[i] for i in range(1, n_ + 1))
    V.report(9, all((3*n_ - 1)*a[n_] + 3*S(n_) == 20 for n_ in range(1, 12)), "원식 n=1..11 성립")
    V.check(9, "1-2 a5/a11", a[5]/a[11], V.ans(9, "1-2"))


def p10():
    a, b = sp.symbols("a b")
    f = x*(x**2 + a*x + b)
    # (x-3)f(x) >= 0 : x^2 (x-3)^2 꼴만 가능 -> 후보 확인
    cand = sp.solve(sp.Poly(sp.expand((x - 3)*f - x**2*(x - 3)**2), x).all_coeffs(), [a, b], dict=True)[0]
    f = sp.expand(f.subs(cand))
    ok = all((xv - 3)*f.subs(x, xv) >= 0 for xv in [sp.Rational(i, 4) for i in range(-40, 60)])
    V.report(10, ok and f == x**3 - 3*x**2, f"f={f}, (x-3)f>=0 격자")
    V.check(10, "1-1 넓이", sp.integrate(abs_ := -f, (x, 0, 3)), V.ans(10, "1-1"))
    ks = [r for r in sp.solve(sp.Eq(f, 16), x) if r.is_real]
    V.check(10, "1-2 k", max(ks), V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
