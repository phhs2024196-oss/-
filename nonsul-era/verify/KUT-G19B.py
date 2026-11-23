#!/usr/bin/env python3
"""KUT-G19B (2019 실전 오후 수학 01~03) 검산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from itertools import combinations
from sympy import Rational as R, symbols, solve, integrate, sqrt, Matrix
V = Verifier("KUT-G19B", "items/KUT/PAST")

def p01():
    a, b, c = symbols("a b c", real=True)
    pts = [(1, 2), (-1, 4), (-3, 2)]
    eqs = [x_**2 + y_**2 + a * x_ + b * y_ + c for x_, y_ in pts]
    # YAML 의 연립방정식을 파싱해 대입 결과와 비교
    given = [e.split("=") for e in V.ans(1, "1-1").split(";")]
    loc = {"a": a, "b": b, "c": c}
    gl = [sp.sympify(l, locals=loc) - sp.sympify(r, locals=loc) for l, r in given]
    V.report(1, all(sp.simplify(g - e) == 0 or sp.simplify(g + e) == 0 for g, e in zip(gl, eqs)), "1-1 연립방정식 = 세 점 대입")
    sol = solve(eqs, [a, b, c], dict=True)[0]
    V.check(1, "1-2 a", sol[a], V.ans(1, "1-2", "a")); V.check(1, "1-2 b", sol[b], V.ans(1, "1-2", "b")); V.check(1, "1-2 c", sol[c], V.ans(1, "1-2", "c"))
    cx, cy = -sol[a] / 2, -sol[b] / 2
    r = sqrt(cx**2 + cy**2 - sol[c])
    V.check(1, "1-3 r", r, V.ans(1, "1-3", "r"))
    V.report(1, str((cx, cy)).replace(" ", "") == V.ans(1, "1-3", "center"), f"1-3 중심 ({cx},{cy})")
    d = sqrt(cx**2 + cy**2) - r
    V.check(1, "1-4", d, V.ans(1, "1-4"))
    # 수치: 원 위 각도 격자에서 원점까지 최소거리
    import math
    m = min(math.hypot(float(cx) + float(r) * math.cos(t * 2 * math.pi / 100000), float(cy) + float(r) * math.sin(t * 2 * math.pi / 100000)) for t in range(100000))
    V.report(1, abs(m - float(sp.N(S(V.ans(1, "1-4"))))) < 1e-6, f"1-4 수치 최소 {m:.6f}")

def p02():
    xx, a, b = symbols("x a b", real=True)
    f = xx**3 + 2 * a * xx**2 + 3 * b * xx + 3
    sol = solve([f.diff(xx).subs(xx, -1), f.subs(xx, -1) + 1], [a, b], dict=True)[0]
    V.check(2, "2-1 a", sol[a], V.ans(2, "2-1", "a")); V.check(2, "2-1 b", sol[b], V.ans(2, "2-1", "b"))
    fx = f.subs(sol)
    crit = solve(fx.diff(xx), xx)
    # x=-1 이 극소인지, 극대값
    V.report(2, fx.diff(xx, 2).subs(xx, -1) > 0, "x=-1 극소")
    mx = [fx.subs(xx, c) for c in crit if fx.diff(xx, 2).subs(xx, c) < 0][0]
    V.check(2, "2-2 극댓값", mx, V.ans(2, "2-2"))
    Sx = integrate(3 - fx, (xx, -3, 0))
    # 교점 확인: f=3 -> x=-3 (접), x=0
    V.report(2, sorted(solve(fx - 3, xx)) == [-3, 0], "곡선과 y=3 교점 x=-3,0")
    V.check(2, "2-3 넓이", Sx, V.ans(2, "2-3"))

def p03():
    cards = [d for d in range(1, 13) if 12 % d == 0]
    assert len(cards) == 6
    hands = list(combinations(cards, 2))
    dist = {x: R(sum(1 for h in hands if sum(1 for c in h if c % 2 == 0) == x), len(hands)) for x in (0, 1, 2)}
    V.check(3, "3-1 pmf x=0", sp.binomial(4, 0) * sp.binomial(2, 2) / sp.binomial(6, 2), dist[0])
    for x_ in (0, 1, 2):
        V.check(3, f"3-2 p{x_}", dist[x_], V.ans(3, "3-2", f"p{x_}"))
    E = sum(x_ * p for x_, p in dist.items())
    V.check(3, "3-3 E", E, V.ans(3, "3-3"))
    Var = sum((x_ - E) ** 2 * p for x_, p in dist.items())
    V.check(3, "3-4 V", Var, R(16, 45))
    V.check(3, "3-4 sigma", sp.sqrt(Var), V.ans(3, "3-4"))
    V.check(3, "3-1 pmf 일치(x=1,2)", [sp.binomial(4, x_) * sp.binomial(2, 2 - x_) / sp.binomial(6, 2) for x_ in (1, 2)], [dist[1], dist[2]])

if __name__ == "__main__":
    V.run(globals())
