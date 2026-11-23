#!/usr/bin/env python3
"""KUT-G25B (2025학년도 실전 오후 01~10) — 정답을 설정으로부터 독립 재계산."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-G25B", "items/KUT/PAST")
R = sp.Rational


def p01():
    a, b = sp.symbols("a b", real=True)
    # 점근선 y=3, b>3 이므로 b-3=3
    bb = sp.solve(sp.Eq(b - 3, 3), b)[0]
    aa = sp.solve(sp.Eq(3 ** (a - 2) + 3, bb), a)[0]
    V.check(1, "a+b", aa + bb, V.ans(1, "1"))


def p02():
    A, B = sp.symbols("A B", real=True)  # A=log2 a, B=log2 b
    line = (B - A) * 3 + A  # x=3 에서의 값
    # 2 = 3B-2A  → b^3/a^2 = 2^(3B-2A)
    Asol = sp.solve(sp.Eq(line, 2), A)[0]          # 2 = 3B-2A 에서 A
    V.check(2, "b^3/a^2", sp.simplify(sp.Integer(2) ** (3 * B - 2 * Asol)), V.ans(2, "1"))


def cnt_roots(k):
    g = x**3 - 12 * x
    r = set(sp.Poly(g - k, x).real_roots()) | set(sp.Poly(g + k, x).real_roots())
    return len(r)


def p03():
    ks = [R(i, 4) for i in range(1, 161)]
    four = [k for k in ks if cnt_roots(k) == 4]
    V.report(3, four == [16] and sp.Integer(four[0]) == sp.Integer(V.ans(3, "1")), f"개수 4인 k(0<k<=40): {four}")


def p04():
    a, b = sp.symbols("a b")
    s = sp.solve([sp.Eq((4 * a + b) / 6, 4), sp.Eq((2 * a + b) / 4, 2 * 2 + 4)], [a, b])
    f = lambda t_: (s[a] * t_ + s[b]) / (t_ + 2)
    V.check(4, "f(15)=f(3)", f(3), V.ans(4, "1"))


def p05():
    a_ = sp.asin(R(2, 5))
    A, Bb, C = a_, sp.pi - a_, 2 * sp.pi + a_
    ok = all(0 <= v < 5 * sp.pi / 2 and sp.simplify(5 * sp.sin(v) - 2) == 0 for v in (A, Bb, C))
    # 해가 정확히 3개임을 수치 격자로 확인(부호 변화)
    import math
    xs = [i * (2.5 * math.pi) / 20000 for i in range(20001)]
    ch = sum(1 for u, v in zip(xs, xs[1:]) if (5 * math.sin(u) - 2) * (5 * math.sin(v) - 2) < 0)
    V.report(5, ok and ch == 3 and same(sp.cos((A + Bb) / 2 + C), V.ans(5, "1")), f"해 {ch}개, 값 {sp.simplify(sp.cos((A + Bb) / 2 + C))}")


def p06():
    a, b, c = sp.symbols("a b c")
    base = 2 * x**3 - 2 * x**2 + a * x + b
    res = []
    for k_ in (2, 3):
        # (x-1)^k 가 f 의 인수 ⇔ 1 이 k 중근
        eqs = [sp.diff(base, x, j).subs(x, 1) for j in range(k_)]
        s = sp.solve(eqs, [a, b], dict=True)
        res.append((k_, s))
    # k=2: 해 존재, k=3: f=2(x-1)^3 과 계수 비교 불가능(−2≠−6)
    k2 = res[0][1][0]
    f2 = sp.factor(base.subs(k2))
    m = sp.limit(f2 / (x - 1) ** 2, x, 1)
    ok = (res[1][1] == []) and same(f2.subs(x, m), V.ans(6, "1"))
    V.report(6, ok, f"k=3 해 {res[1][1]}, f={f2}, m={m}, f(m)={f2.subs(x, m)}")


def p07():
    a, b = sp.symbols("a b")
    f = lambda t_: t_**2 + a * t_ + b
    gl = sp.expand(f(x) - f(x - 1))
    gr = sp.expand(f(x))
    s = sp.solve([gl.subs(x, 0) - gr.subs(x, 0), sp.diff(gl, x).subs(x, 0) - sp.diff(gr, x).subs(x, 0)], [a, b])
    V.check(7, "f(3)", f(3).subs(s), V.ans(7, "1"))


def p08():
    cs = sp.symbols("c0:5")
    f = sum(c * x**i for i, c in enumerate(cs))
    eq = sp.expand(4 * x**2 - 2 * f - (x + 1 + f.subs(x, -x)))
    s = sp.solve(sp.Poly(eq, x).all_coeffs(), cs, dict=True)[0]
    V.check(8, "∫f", sp.integrate(f.subs(s), (x, 0, 1)), V.ans(8, "1"))


def p09():
    t_ = sp.symbols("t")
    F = lambda u: sp.integrate(t_ * (t_ - 2) * (t_ - 4), (t_, 0, u))
    fx = F(x)
    crit = sp.solve(sp.diff(fx, x), x)
    mx = [fx.subs(x, c) for c in crit if sp.diff(fx, x, 2).subs(x, c) < 0]
    V.check(9, "9-1 극댓값", mx[0], V.ans(9, "9-1"))
    g = sp.expand(fx.subs(x, x + 2) - fx)
    cg = sp.solve(sp.diff(g, x), x)
    mn = [g.subs(x, c) for c in cg if sp.diff(g, x, 2).subs(x, c) > 0]
    V.check(9, "9-2 극솟값", mn[0], V.ans(9, "9-2"))


def p10():
    sols = []
    for d in range(1, 80):
        for a1 in range(-300, 0):
            N = 400
            a = [a1 + d * i for i in range(N)]
            if any(v == 0 for v in a):
                continue
            S = [sum(a[:i + 1]) for i in range(N)]
            if len(set(S)) != N:
                continue
            nmin = next(i + 1 for i, v in enumerate(S) if v > 0)
            if nmin != 13:
                continue
            M = sorted(S)
            if M[1] - M[0] == 2 and M[2] - M[1] == 2:
                sols.append((d, a1))
    V.report(10, sols == [(6, -34)] and str(sols[0][0]) == V.ans(10, "10-1") and str(sols[0][1]) == V.ans(10, "10-2"), f"해 (d,a1)={sols}")


if __name__ == "__main__":
    V.run(globals())
