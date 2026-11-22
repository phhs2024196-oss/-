#!/usr/bin/env python3
"""CUK-M23A (가톨릭대 2023학년도 모의논술 자연공학계열·간호학과 문항 1~3) — 문제 조건에서 독립 재계산."""
import sys, re, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-M23A", "items/CUK/PAST")
mp.mp.dps = 30

def p01():
    a, b, c = sp.symbols("a b c", real=True)
    f = lambda X: (b * X + c) / (X + a)
    # (1) f(13)=2, (2) 직선 y=x-3 대칭: (X,f(X)) -> (f(X)+3, X-3) 도 그래프 위, (3) y=-x+5: (X,Y)->(5-Y,5-X)
    eqs = [f(13) - 2]
    for X in (0, 1, 2, 5, 7):
        eqs.append(sp.together(f(f(X) + 3) - (X - 3)))
        eqs.append(sp.together(f(5 - f(X)) - (5 - X)))
    eqs = [sp.numer(sp.together(e)) for e in eqs]
    sols = sp.solve(eqs, [a, b, c], dict=True)
    sols = [s for s in sols if len(s) == 3 and all(sp.simplify(sp.denom(sp.together(f(X).subs(s)))) != 0 for X in (0, 1, 2, 5, 7))]
    V.report(1, len(sols) >= 1, f"1-1 해 {sols}")
    s = sols[0]
    V.check(1, "1-1 a", s[a], V.ans(1, "1", "a")); V.check(1, "1-1 b", s[b], V.ans(1, "1", "b")); V.check(1, "1-1 c", s[c], V.ans(1, "1", "c"))
    # 논제 2: 점근선 x=-a=4, y=b=1 ; 곡선 위 점 P 에서 두 점근선까지 거리 합의 최소(수치)
    F = lambda X: (s[b] * X + s[c]) / (X + s[a])
    ax_, ay_ = -s[a], s[b]
    g = sp.lambdify(x, abs(x - ax_) + abs(F(x) - ay_), "mpmath")
    best = min(g(mp.mpf(i) / 1000 + mp.mpf("0.0000001")) for i in range(-30000, 60000) if abs(mp.mpf(i) / 1000 - ax_) > 1e-6)
    V.check(1, "1-2 최솟값", sp.Float(best, 12), V.ans(1, "2"), 1e-6)

def p02():
    Nn = sp.Symbol("N")
    out = []
    for N in range(1, 11):
        f = (-1)**N * (x**3 - 3 * x**2 + 3 * x - 12 * N**4 * x + 2)
        crit = sp.solve(sp.diff(f, x), x)
        mx = [cc for cc in crit if sp.diff(f, x, 2).subs(x, cc) < 0]
        assert len(mx) == 1
        out.append(mx[0])
    ans = sp.sympify(V.ans(2, "1"), locals={"n": n})
    V.report(2, all(sp.simplify(ans.subs(n, N) - out[N - 1]) == 0 for N in range(1, 11)), "2-1 a_n=극대 위치 (n=1..10)")
    V.check(2, "2-2 S", sum(out), V.ans(2, "2"))

def p03():
    a = sp.Symbol("a", real=True)
    f = x**3 + x**2 + 5 * x - 7; g = 2 * x**2 - x - 1
    h = f - a * g
    nums = sorted({sp.Rational(v) for v in re.findall(r"-?\d+(?:/\d+)?", V.ans(3, "1"))})
    p1, p2, p3_ = sorted(nums)
    in_A = lambda av: av < p1 or (p2 < av < p3_) or av > p3_
    bad = []
    for i in range(-600, 601):
        av = sp.Rational(i, 60)
        cnt = len(set(sp.real_roots(sp.Poly(h.subs(a, av), x))))
        if (cnt == 3) != in_A(av): bad.append(av)
    V.report(3, not bad, f"3-1 A 격자 판정 불일치 {bad[:3]}")
    # 논제 2: 중근 a 값 = h 의 판별식=0 이고 삼중근 아닌 a
    disc = sp.discriminant(sp.expand(h), x)
    av_set = [r for r in sp.solve(disc, a) if len(set(sp.real_roots(sp.Poly(h.subs(a, r), x)))) == 2]
    vals = []
    for r in av_set:
        rts = sorted(sp.real_roots(sp.Poly(h.subs(a, r), x)))
        al, be = rts[0], rts[-1]
        gl = sp.lambdify(x, g, "mpmath")
        pts = [al] + [q for q in (-sp.Rational(1, 2), 1) if al < q < be] + [be]
        vals.append(mp.quad(lambda t: abs(gl(t)), [float(q) for q in pts]))
    V.report(3, sorted(av_set) == [-2, 3, sp.Rational(10, 3)], f"3-2 중근 a {av_set}")
    V.check(3, "3-2 min", sp.Float(min(vals), 15), V.ans(3, "2"), 1e-9)

if __name__ == "__main__": V.run(globals())
