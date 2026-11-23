#!/usr/bin/env python3
"""KUT-G19A (2019 실전 오전 수학 01~03) 검산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from sympy import Rational as R, symbols, solve, integrate, Abs, Piecewise, sqrt
V = Verifier("KUT-G19A", "items/KUT/PAST")

def p01():
    xx, y = symbols("x y", real=True)
    # 닮음: 삼각형 APS (밑변 x, 높이 80-y) ~ ABC (밑변 100, 높이 80)
    yy = solve(sp.Eq(xx / 100, (80 - y) / 80), y)[0]
    V.check(1, "1-1 y=f(x)", yy, sp.sympify(V.ans(1, "1-1"), locals={"x": xx}))
    G = xx * yy
    V.check(1, "1-2 G(x)", G, sp.sympify(V.ans(1, "1-2"), locals={"x": xx}))
    mn, mx = extrema(G, xx, 0, 100)
    V.check(1, "1-3 max", mx, V.ans(1, "1-3"))
    xs = solve(G.diff(xx), xx)[0]
    V.check(1, "1-4 x", xs, V.ans(1, "1-4", "x")); V.check(1, "1-4 y", yy.subs(xx, xs), V.ans(1, "1-4", "y"))
    # 격자 탐색으로 최대 확인
    best = max(((R(t, 2) * (80 - R(4, 5) * R(t, 2))), R(t, 2)) for t in range(0, 201))
    V.check(1, "1-3 격자 최대", best[0], V.ans(1, "1-3"))

def p02():
    xx, a = symbols("x a", real=True)
    f1 = (xx + 1) * (xx - a); f2 = 4 * (xx - a) * (2 - xx)
    Sa = -integrate(f1, (xx, -1, a)) + integrate(f2, (xx, a, 2))
    V.check(2, "2-1 S(a)", sp.expand(Sa), sp.expand(sp.sympify(V.ans(2, "2-1"), locals={"a": a})))
    # 부호 확인: f1<=0 on [-1,a], f2>=0 on [a,2] (a 표본)
    for av in [R(-1), R(-1, 2), R(0), R(1), R(2)]:
        if av > -1: assert all(f1.subs({a: av, xx: av - (av + 1) * R(t, 10)}) <= 0 for t in range(0, 11))
        if av < 2: assert all(f2.subs({a: av, xx: av + (2 - av) * R(t, 10)}) >= 0 for t in range(0, 11))
    # 수치 정의(|f| 적분)와 비교
    for av in [R(-1), R(0), R(1), R(2)]:
        num = sp.N(sp.Integral(Abs(Piecewise((f1.subs(a, av), xx <= av), (f2.subs(a, av), True))), (xx, -1, 2)), 15)
        assert abs(num - sp.N(Sa.subs(a, av), 15)) < 1e-6, (av, num)
    ex = solve(Sa.diff(a), a)
    ex_in = [r for r in ex if -1 < r < 2]
    V.check(2, "2-2 극값점", ex_in[0], V.ans(2, "2-2"))
    mn, mx = extrema(Sa, a, -1, 2)
    V.check(2, "2-3 max", mx, V.ans(2, "2-3", "max")); V.check(2, "2-3 min", mn, V.ans(2, "2-3", "min"))
    V.report(2, (Sa.subs(a, 2) == R(9, 2)), "S(2)=9/2 (예시답안)")

def p03():
    # 3x4 키패드, 상하좌우 이웃마다 1/8 로 오타
    pad = {'1': (0, 0), '2': (0, 1), '3': (0, 2), '4': (1, 0), '5': (1, 1), '6': (1, 2), '7': (2, 0), '8': (2, 1), '9': (2, 2), '*': (3, 0), '0': (3, 1), '#': (3, 2)}
    pos = {v: k for k, v in pad.items()}
    def nbrs(c):
        r, q = pad[c]
        return [pos[(r + dr, q + dc)] for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, q + dc) in pos]
    def pexact(c): return 1 - R(len(nbrs(c)), 8)
    V.check(3, "3-1", pexact('2'), V.ans(3, "3-1"))
    V.check(3, "3-2", pexact('1'), V.ans(3, "3-2"))
    V.check(3, "3-3", pexact('5') * pexact('7'), V.ans(3, "3-3"))
    # 전수: 1,0,6 을 시도할 때 실제 눌린 결과 분포에서 106 이 나올 확률
    def dist(c):
        d = {c: pexact(c)}
        for n in nbrs(c): d[n] = R(1, 8)
        return d
    p106 = sum(R(1) * dist('1').get(x, 0) * dist('0').get(y, 0) * dist('6').get(z, 0) for x in '1' for y in '0' for z in '6')
    tot = sum(dist('1')[x] * dist('0')[y] * dist('6')[z] for x in dist('1') for y in dist('0') for z in dist('6'))
    V.check(3, "3-4 분포 합=1", tot, 1)
    V.check(3, "3-4", 1 - p106, V.ans(3, "3-4"))

if __name__ == "__main__":
    V.run(globals())
