#!/usr/bin/env python3
"""KUT-G23A-01~10 (2023학년도 한국기술교육대 실전 오전) — 문제 설정으로부터 독립 재계산 후 YAML 답과 비교."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
import math
import random
V = Verifier("KUT-G23A", "items/KUT/PAST")
R = sp.Rational


def p01():
    a, b = sp.solve(x**2 - 4*x + 2, x)
    A, B = sp.sqrt(a), sp.sqrt(b)
    V.check(1, "식의 값", sp.nsimplify(sp.N((A - 1/A)**2 + (B - 1/B)**2, 40), rational=True, tolerance=1e-30), V.ans(1, "1"))


def p02():
    f = lambda m: sp.log(2**m, 8) - 1
    g = lambda m: sp.log(2**m, 2) + 1
    ms = [m for m in range(1, 400) if m != 3
          and (lambda v: v > 0 and v == int(v))(sp.simplify(g(m)/f(m) - 3))]
    V.check(2, f"조건 만족 m={ms} 의 합", sum(ms), V.ans(2, "1"))


def p03():
    a = sp.Symbol("a")
    f = (x + 1)*(x - a)
    sol = [s for s in sp.solve(sp.Eq(sp.limit(2*(x**4 - 1)/((x - 1)*f), x, 1), 1), a)]
    f0 = f.subs(a, sol[0])
    V.check(3, f"a={sol} f'(2)", sp.diff(f0, x).subs(x, 2), V.ans(3, "1"))


def p04():
    xs = [R(k, 7) + R(1, 3) for k in range(1, 12)]
    cnt = 0
    for m in range(1, 11):
        for n in range(1, 11):
            ok = all(abs(sp.N(sp.sin(v) + sp.sin(v - sp.Rational(n, m)*sp.pi), 30)) < 1e-20 for v in xs)
            cnt += ok
    V.check(4, "항등 성립 순서쌍 수", cnt, V.ans(4, "1"))


def p05():
    a = sp.Symbol("a", positive=True)
    f = a*x**3 - 4*x**2 + a*x + 6
    D4 = sp.discriminant(sp.diff(f, x), x)
    amin = min(s for s in sp.solve(D4, a) if s > 0)
    ok_hold = all(sp.diff(f, x).subs({a: amin, x: v}) >= -1e-12 for v in [R(i, 10) for i in range(-50, 51)])
    ok_fail = any(sp.diff(f, x).subs({a: amin - R(1, 100), x: v}) < 0 for v in [R(i, 100) for i in range(0, 200)])
    V.report(5, ok_hold and ok_fail, f"a_min={amin} 에서 f'>=0 유지, 그보다 작으면 f'<0 인 x 존재")
    V.check(5, "a 최솟값", amin, V.ans(5, "1"))


def p06():
    c = sp.symbols("c0:4")
    f = sum(c[i]*x**i for i in range(4))
    eqs = [c[3] - 2, f.subs(x, 2), f.subs(x, 1), sp.diff(f, x).subs(x, 1)/5 - R(4, 5)]
    s = sp.solve(eqs, c, dict=True)[0]
    f = f.subs(s)
    V.check(6, "조건 (다) 극한", sp.limit(f/(4*x**2 - 3*x - 1), x, 1), R(4, 5))
    V.check(6, "f(4)", f.subs(x, 4), V.ans(6, "1"))


def p07():
    c = sp.symbols("c0:5")
    f = sum(c[i]*x**i for i in range(5))
    t = sp.Symbol("t")
    rhs = x**2 + sp.integrate(((t - 1)*sp.diff(f, x)).subs(x, t), (t, 0, x))
    s = sp.solve(sp.Poly(sp.expand(x*f - rhs), x).all_coeffs(), c, dict=True)[0]
    f = f.subs(s)
    V.check(7, "∫_{-1}^{2} f", sp.integrate(f, (x, -1, 2)), V.ans(7, "1"))


def p08():
    t = sp.Symbol("t", real=True)
    f = 6*t - 3
    g = 3*t**2 - 4*t - 4
    xP = sp.integrate(f, (t, 0, t)) if False else sp.integrate(f, t)
    xP = xP - xP.subs(t, 0)
    xQ = 5 + sp.integrate(g, t) - sp.integrate(g, t).subs(t, 0)
    rts = sorted(r for r in sp.solve(sp.Eq(xP, xQ), t) if r > 0)
    al, be = rts[0], rts[1]

    def dist(v):
        zs = [r for r in sp.solve(v, t) if al < r < be]
        pts = [al] + sorted(zs) + [be]
        return sum(abs(sp.integrate(v, (t, p, q))) for p, q in zip(pts, pts[1:]))
    V.check(8, f"α={al}, β={be}, s2-s1", dist(g) - dist(f), V.ans(8, "1"))


def p09():
    t = sp.Symbol("t")
    g = (-(x**3 - 6*x**2 + 7*x) + (-2*x + 6))
    mn, mx = extrema(g, x, 0, 3)
    V.check(9, "1-1 a = max g", mx, V.ans(9, "1-1"))
    # t=mn 에서 한 점, t 가 (mn, mx) 에서는 두 점, 값 확인
    cnt = lambda tv: len([r for r in sp.Poly(g - tv, x).nroots() if abs(sp.im(r)) < 1e-9 and -1e-9 <= sp.re(r) <= 3 + 1e-9])
    V.report(9, cnt(mn) == 1 or cnt(mn) == 2, f"t=min 일 때 교점수 {cnt(mn)}(이중근 1점으로 셈)")
    b = mn
    V.check(9, "1-2 f(b)", (x**3 - 6*x**2 + 7*x + b).subs(x, b), V.ans(9, "1-2"))


def p10():
    sols = []
    for a1 in range(-80, 81):
        for b1 in range(-80, 81):
            a = lambda n: a1 + 3*(n - 1)
            b = lambda n: b1 + 4*(n - 1)
            if not any(a(k) == 0 for k in range(1, 9)):
                continue
            if a1 < b1 and 2*a(5) + b(6) == 2 and sum(abs(b(n)) for n in range(1, 9)) == sum(abs(a(n)) for n in range(1, 9)) + 14:
                sols.append((a1, b1))
    V.report(10, len(sols) == 1, f"조건을 모두 만족하는 (a1,b1) = {sols}")
    a1, b1 = sols[0]
    V.check(10, "1-1 a1+b1", a1 + b1, V.ans(10, "1-1"))
    V.check(10, "1-2 합", sum((b1 + 4*(n - 1)) - (a1 + 3*(n - 1)) for n in range(1, 11)), V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
