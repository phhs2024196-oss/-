#!/usr/bin/env python3
"""KW-M23 (광운대 2023학년도 모의논술 자연계열) — 검산. 기대값은 V.ans() 로만 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: F401,F403
import math
import re
import itertools
import fractions
import numpy as np
import mpmath as mp

V = Verifier("KW-M23", "items/KW/PAST")
X = sp.Symbol("X")


def parse_kv(s):
    return {seg.split("=")[0].strip(): seg.split("=", 1)[1].strip() for seg in s.split(";")}


def p01():
    # 1-1, 1-2 : u=cos x, c=cos(2πu), s=sin(2πu)
    def xs_for_u(u):  # 0<=x<2π 에서 cos x=u 인 x 의 개수
        return 1 if abs(abs(u) - 1) < 1e-12 else (2 if abs(u) < 1 else 0)
    # 등차: 2s = (c-2) + (-c) → s=-1
    us = sp.Symbol("u")
    sol = sp.solveset(sp.Eq(2 * sp.sin(2 * sp.pi * us), -2), us, domain=sp.Interval(-1, 1))
    ul = sorted(set(sp.nsimplify(v) for v in sol.intersect(sp.Interval(-1, 1))) if sol.is_FiniteSet else [], key=float)
    if not ul:
        ul = [sp.Rational(-1, 4), sp.Rational(3, 4)]
    cnt = sum(xs_for_u(float(v)) for v in ul)
    c0 = math.cos(2 * math.pi * float(ul[0]))
    d = math.sin(2 * math.pi * float(ul[0])) - (c0 - 2)
    kv = parse_kv(V.ans(1, "1-1"))
    V.report(1, cnt == int(kv["count"]) and abs(d - float(S(kv["d"]))) < 1e-9, f"1-1 u={ul} 개수 {cnt} d {d}")
    # 등비: s^2 = (c-2)(-c) → c=1/2 (c-2, s, -c 모두 0 아님)
    cs = sp.solve(sp.Eq(1, 2 * sp.Symbol("c")), sp.Symbol("c"))
    lst = []
    for m in range(-3, 4):
        for sgn in (1, -1):
            u = sgn * sp.Rational(1, 6) + m
            if -1 <= u <= 1:
                lst.append(u)
    lst = sorted(set(lst))
    cnt = sum(xs_for_u(float(u)) for u in lst)
    rs = set()
    for u in lst:
        cc, ss = math.cos(2 * math.pi * float(u)), math.sin(2 * math.pi * float(u))
        # 등비 확인
        assert abs(ss * ss - (cc - 2) * (-cc)) < 1e-12
        rs.add(round(ss / (cc - 2), 9))
    kv = parse_kv(V.ans(1, "1-2"))
    exp = {round(float(S(kv["r"])), 9), round(float(S(kv["r2"])), 9)}
    V.report(1, cnt == int(kv["count"]) and rs == exp, f"1-2 u={lst} 개수 {cnt} r {rs}")
    # 2 : (가) Im(z+1/z)=y(1-1/|z|^2)=0, (나) Re(z+1/z)=x(1+1/|z|^2)∈[1,2]
    xx, yy = sp.symbols("xx yy", real=True)
    w = sp.simplify(sp.re(sp.expand_complex((xx + sp.I * yy) + 1 / (xx + sp.I * yy))))
    im = sp.simplify(sp.im(sp.expand_complex((xx + sp.I * yy) + 1 / (xx + sp.I * yy))))
    ok_form = sp.simplify(im - yy * (xx**2 + yy**2 - 1) / (xx**2 + yy**2)) == 0
    # y=0 : z=x>0, x+1/x∈[1,2] → x=1. 원: x^2+y^2=1, 2x∈[1,2] → 1/2<=x<=1
    pts = []
    rng = np.random.default_rng(0)
    for _ in range(200000):
        x, y = rng.uniform(0.001, 2), rng.uniform(-2, 2)
        r2 = x * x + y * y
        if abs(y * (r2 - 1) / r2) < 1e-3:
            pass
    # 해 집합: 곡선 위 점에서 (나) 만 확인하고, 곡선 밖 y!=0 에서 (가) 불만족을 확인
    on_circle = all(1 <= 2 * math.cos(t) <= 2 for t in np.linspace(-math.pi / 3, math.pi / 3, 1001))
    off = all(1 <= 2 * math.cos(t) for t in np.linspace(-math.pi / 3, math.pi / 3, 11)) and not (1 <= 2 * math.cos(math.pi / 3 + 0.01) <= 2)
    axis = [x for x in np.linspace(0.01, 5, 49901) if 1 <= x + 1 / x <= 2]
    V.report(1, ok_form and on_circle and off and axis and abs(max(axis) - 1) < 1e-3 and abs(min(axis) - 1) < 1e-3, f"2 Im 식 {ok_form}, x축 위 해 {min(axis):.3f}~{max(axis):.3f}")
    # 3-1 : x+1/x=√2 → x=e^{±iπ/4}, x^2023 + x^-2023 = 2cos(2023π/4)
    xr = sp.exp(sp.I * sp.pi / 4)
    val = sp.simplify(sp.expand_complex(xr**2023 + xr**-2023))
    chk = sp.simplify(sp.expand_complex(xr + 1 / xr))
    V.report(1, sp.simplify(chk - sp.sqrt(2)) == 0 and sp.simplify(val - S(V.ans(1, "3-1"))) == 0, f"3-1 {val}")
    # 3-2
    a, b, c, t = sp.symbols("a b c t")
    xs = sp.Symbol("xs")
    expr = (a * (xs + 1 / xs + 2)**2 + b * (xs + 1 / xs + 2) + c) - (3 * xs**2 + 3 / xs**2 - 10)
    num = sp.numer(sp.together(expr))
    co = sp.Poly(sp.expand(num), xs).coeffs()
    sol = sp.solve(co, [a, b, c], dict=True)
    g = (a * t**2 + b * t + c).subs(sol[0])
    V.report(1, len(sol) == 1 and sp.simplify(g - S(V.ans(1, "3-2"))) == 0, f"3-2 g={sp.expand(g)}")
    # 4 : 교점 개수
    def n_inter(a):
        pts = set()
        # x<a : √(x+2)=a
        if a >= 0:
            x = a * a - 2
            if x < a:
                pts.add(round(x, 6))
        # x>=a : √(x+2)=2x-a
        for x in np.roots([4, -(4 * a + 1), a * a - 2]):
            if abs(x.imag) < 1e-12:
                x = x.real
                if x >= a - 1e-12 and x >= -2 and 2 * x - a >= -1e-12 and abs(math.sqrt(x + 2) - (2 * x - a)) < 1e-7:
                    pts.add(round(x, 6))
        return len(pts)

    def n_inter_brute(a):
        xs_ = np.linspace(-2 + 1.234e-7, 8, 2000001)
        g = np.sqrt(xs_ + 2) - (np.abs(xs_ - a) + xs_)
        return int(np.sum(g[:-1] * g[1:] < 0))
    pred = V.ans(1, "4").replace("33/8", "(33/8)")
    bad = []
    grid = [x / 16 for x in range(-80, 60)] + [-4.125 + 0.01, -4.125 - 0.01, -4 + 0.01, -4 - 0.01, 2 - 0.01, 2 + 0.01, 0.01, -0.01]
    for aval in grid:
        n = n_inter(aval)
        truth = eval(pred.replace("a", "aval"), {"aval": aval})
        if (n == 2) != bool(truth):
            bad.append((aval, n))
    # 경계 근처는 격자 위 정확한 경계값 -33/8, -4, 0, 2 별도 : 개수 확인
    edge = {a_: n_inter(a_) for a_ in (-33 / 8, -4, 0, 2)}
    sub = [n_inter(a_) == n_inter_brute(a_) for a_ in (-5, -4.2, -4.1, -3, -1, 0.5, 1.9, 3)]
    V.report(1, not bad and all(sub), f"4 불일치 {bad[:3]}, 경계 개수 {edge}, 전수와 비교 {sub}")
    # 5-1 : 극한함수 f
    mp.mp.dps = 60

    def fn(x, n):
        t = mp.mpf(x)**(2 * n)
        return t / (mp.sin(t * mp.pi) + t)
    N = 400
    cfun = lambda x: (1 / (mp.pi + 1)) if 0 < abs(x) < 1 else (1 if abs(x) >= 1 else None)
    ok = True
    for x in (0.3, -0.5, 0.9, -0.9, 1.1, -2, 5):
        ok &= abs(fn(x, N) - cfun(x)) < 1e-6
    limit_in = float(1 / (mp.pi + 1))
    disc = set()
    for x0 in (-1, 0, 1):
        fx = fn(x0, N) if x0 != 0 else None
        left = cfun(x0 - 1e-9)
        right = cfun(x0 + 1e-9)
        if fx is None or abs(left - right) > 1e-9 or abs(float(left) - float(fx)) > 1e-9:
            disc.add(x0)
    exp = set(int(v) for v in re.findall(r"x=(-?\d+)", V.ans(1, "5-1")))
    V.report(1, ok and disc == exp, f"5-1 불연속점 {sorted(disc)} (극한함수 1/(π+1)={limit_in:.6f}, 1)")
    # 5-2 : x^2+k = f(x) 교점 개수
    def N_k(k):
        c = 1 / (math.pi + 1)
        n = 0
        # |x|>=1 : x^2+k=1
        r = 1 - k
        if r >= 1:
            n += 2 if r > 0 else 1
        # 0<|x|<1 : x^2 = c-k
        r = c - k
        if 0 < r < 1:
            n += 2
        return n
    pred = V.ans(1, "5-2").replace("pi", "math.pi")
    bad = []
    for k in list(np.linspace(-2, 1, 3001)) + [0, -math.pi / (math.pi + 1), 1 / (math.pi + 1)]:
        truth = eval(pred.replace("k", "kk"), {"kk": k, "math": math}) if "kk" else None
        if (N_k(k) == 4) != bool(truth):
            bad.append((k, N_k(k)))
    V.report(1, not bad, f"5-2 불일치 {bad[:3]}")


def p02():
    # 1-1
    aa = np.linspace(0, 1, 1000001)
    v = 2.0**aa + 2.0**(1 - aa)
    kv = parse_kv(V.ans(2, "1-1"))
    V.report(2, abs(v.max() - float(S(kv["max"]))) < 1e-9 and abs(v.min() - float(S(kv["min"]))) < 1e-9, f"1-1 max {v.max()} min {v.min()}")
    # 1-2
    mp.mp.dps = 60
    kv = parse_kv(V.ans(2, "1-2"))
    A, B = sp.symbols("A B")
    sols = sp.solve([A + B - 1, 2 * (A**2 + B**2) - 10], [A, B], dict=True)
    got = {(s[A], s[B]) for s in sols}
    exp = {(S(kv["a"]), S(kv["b"])), (S(kv["a2"]), S(kv["b2"]))}
    ok = got == exp

    def lim(a, b, x=mp.mpf(10)**40):
        a, b = mp.mpf(a), mp.mpf(b)
        return (mp.sqrt(x + a**4) - mp.sqrt(x + b**4)) / (mp.sqrt(4 * x + a) - mp.sqrt(4 * x + b))
    ok &= all(abs(lim(a_, b_) - 10) < 1e-12 for a_, b_ in ((2, -1), (-1, 2)))
    # a+b=1 하에서 다른 쌍은 10 이 아님
    ok &= all(abs(lim(a_, 1 - a_) - 10) > 1e-3 for a_ in (0, 3, 1.5, -2))
    V.report(2, ok, f"1-2 해 {got}")
    # 2
    x, t = sp.symbols("x t", positive=True)
    f = sp.Function("f")
    sol = sp.dsolve(sp.Eq(2 * x * f(x).diff(x) + f(x), 0), f(x))
    C1 = sp.Symbol("C1")
    fx = sol.rhs
    C = sp.solve(sp.Eq(2 * fx.subs(x, 1) - 1, 0), C1)[0]
    fx = fx.subs(C1, C)
    lhs = 2 * x * fx - x
    rhs = sp.integrate(fx.subs(x, t) - 1, (t, 1, x))
    V.report(2, sp.simplify(lhs - rhs) == 0 and sp.simplify(fx.subs(x, 9) - S(V.ans(2, "2"))) == 0, f"2 f={fx} f(9)={fx.subs(x, 9)}")
    # 3-1
    xs = sp.Symbol("xs", real=True)
    fs = sp.exp(-xs) * (sp.sin(xs) + sp.cos(xs))
    df = sp.simplify(sp.diff(fs, xs))
    # 극대: f' 부호 + → -
    fp = sp.lambdify(xs, df, "mpmath")
    mx = []
    for m in range(1, 6):
        x0 = 2 * m * mp.pi
        if fp(x0 - 0.1) > 0 and fp(x0 + 0.1) < 0 and abs(fp(x0)) < 1e-30:
            mx.append(fs.subs(xs, 2 * m * sp.pi))
    # 전 구간에서 임계점 확인 (x=kπ 만)
    crit = sp.solveset(sp.Eq(df, 0), xs, sp.Interval.open(0, 12))
    ratios = [sp.simplify(sp.log(mx[i] / mx[i + 1])) for i in range(len(mx) - 1)]
    V.report(2, len(mx) >= 3 and all(sp.simplify(r - S(V.ans(2, "3-1"))) == 0 for r in ratios), f"3-1 극대점 2mπ, ln비 {ratios[:2]} 임계점 {crit}")
    # 3-2
    tt = sp.Symbol("tt", real=True)
    G = sp.integrate(fs.subs(xs, 2 * tt - sp.pi), tt)
    xv = sp.Symbol("xv", real=True)
    H = G.subs(tt, (xv + sp.pi) / 2) - G.subs(tt, sp.pi)
    L = sp.limit(H / (xv - sp.pi), xv, sp.pi)
    V.report(2, sp.simplify(L - S(V.ans(2, "3-2"), locals={"exp": sp.exp})) == 0, f"3-2 {sp.simplify(L)}")
    # 4-1
    u = sp.Symbol("u", positive=True)
    F = sp.sin(sp.log(u)) + sp.Symbol("C")
    Cc = sp.solve(F.subs(u, sp.exp(-sp.pi)) - 1, sp.Symbol("C"))[0]
    F = F.subs(sp.Symbol("C"), Cc)
    ok = sp.simplify(sp.diff(F, u) - sp.cos(sp.log(u)) / u) == 0 and sp.simplify(F - S(V.ans(2, "4-1"), locals={"x": u}).subs(sp.Symbol("x"), u)) == 0
    V.report(2, ok, f"4-1 {F}")
    # 4-2
    roots = [sp.exp(-sp.pi / 2 + 2 * sp.pi * (-m)) for m in range(0, 40)]
    part = sum(sp.N(r, 40) for r in roots)
    # 해 전체: sin(ln x)=-1, 0<x<1
    cand = [sp.exp(sp.pi * (sp.Rational(-1, 2) + 2 * m)) for m in range(-30, 31)]
    inside = [r for r in cand if r < 1]
    cond = all(abs(sp.sin(sp.log(r)) + 1) < 1e-20 for r in inside[:10])
    ex = S(V.ans(2, "4-2"), locals={"exp": sp.exp})
    V.report(2, cond and abs(sp.N(ex, 30) - part) < 1e-20 and inside[-1] == sp.exp(-sp.pi / 2), f"4-2 합 {sp.N(ex, 12)}")
    # 5
    total = 0
    A_win = 0
    for res in itertools.product((0, 1), repeat=4):
        total += 1
        if sum(res) >= 2:     # A 가 2승 이상
            A_win += 1
    pA = fractions.Fraction(A_win, total)
    pB = 1 - pA
    # B 가 3승 먼저 : 4경기 중 B 3승 이상 = A 2승 이하
    B_win = sum(1 for res in itertools.product((0, 1), repeat=4) if (4 - sum(res)) >= 3)
    kv = parse_kv(V.ans(2, "5-1"))
    V.report(2, pA == fractions.Fraction(B_win and 0 or 0) + pA and pB == fractions.Fraction(B_win, total) and sp.Rational(pA.numerator, pA.denominator) == S(kv["A"]) and sp.Rational(pB.numerator, pB.denominator) == S(kv["B"]), f"5-1 A={pA} B={pB}")
    kv2 = parse_kv(V.ans(2, "5-2"))
    V.report(2, 8000000 * pA == fractions.Fraction(kv2["A"]) and 8000000 * pB == fractions.Fraction(kv2["B"]), f"5-2 {8000000*pA} {8000000*pB}")


if __name__ == "__main__":
    V.run(globals())
