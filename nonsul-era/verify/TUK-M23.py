#!/usr/bin/env python3
"""TUK-M23-01~03 (한국공학대 2023 모의논술) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-M23", "items/TUK/PAST")


def E(s):
    return sp.sympify(s, locals={"x": x, "n": n, "t": t})


def p01():
    N = n
    S = sp.Rational(3, 2) * (N + N**2)
    a = sp.simplify(S - S.subs(N, N - 1))
    V.check(1, "a_n", a, E(V.ans(1, "1")))
    c = sp.simplify((a.subs(N, N + 1) / 3) ** 2 - (a / 3) ** 2 - 1)
    V.check(1, "sum c_n (1..10)", sum(c.subs(N, i) for i in range(1, 11)), V.ans(1, "2"))
    an = lambda i: 3 * i  # noqa: E731
    f = sum(sp.Rational(1, 7) * (x - an(i)) ** 2 for i in range(1, 8))
    V.check(1, "min f", extrema(sp.expand(f), x, -50, 50)[0], V.ans(1, "3"))
    X, Y = sp.symbols("X Y")
    tot = 0
    for i in range(1, 6):
        sol = sp.solve([X + 3 * Y - an(i) ** 2, X - Y / i - 1], [X, Y])
        A = (sol[X], sol[Y])
        B = (0, sp.Rational(an(i) ** 2, 3))  # x=0 에서 3y=a^2
        C = (0, sp.Integer(-i))  # x=0 에서 -y/n=1
        area = sp.Abs(B[1] - C[1]) * sp.Abs(A[0]) / 2
        tot += area / 3
    V.check(1, "sum M_n/3", tot, V.ans(1, "4"))


def p02():
    c, d = sp.symbols("c d")
    f2 = 6 * x - x**2 + d
    dd = sp.solve(f2.subs(x, 2), d)[0]
    f1 = f2.subs(d, dd).subs(x, 1)
    V.check(2, "f(1)", f1, V.ans(2, "1"))
    cc = sp.solve(4 * 1 + c - f1, c)[0]
    g1, g2 = 4 * x + cc, f2.subs(d, dd)
    V.check(2, "max f", max(extrema(g1, x, -50, 1)[1], extrema(g2, x, 1, 50)[1]), V.ans(2, "2"))
    lim = sp.limit((3 * g2 - x * g2.subs(x, 3)) / (x**2 - 9), x, 3)
    V.check(2, "limit", lim, V.ans(2, "3"))
    val = sp.integrate(sp.Piecewise((0, x <= 2), (2 * g2, True)) , (x, 0, 3))
    V.check(2, "integral f+|f|", val, V.ans(2, "4"))
    # 수치 확인
    num = sp.integrate(sp.Abs(g1) + g1, (x, 0, 1)) + sp.integrate(sp.Abs(g2) + g2, (x, 1, 3))
    V.check(2, "integral(수치 재확인)", sp.N(num, 20), V.ans(2, "4"), 1e-9)


def p03():
    th = sp.Symbol("th", real=True)
    f = lambda T: 3 * x**3 + 6 * sp.sin(T) * sp.cos(T) * x**2 + sp.sin(T) ** 2 * x + sp.cos(T)  # noqa: E731
    V.check(3, "root at pi/2", sp.solve(f(sp.pi / 2), x)[0], V.ans(3, "1"))
    tan = sp.expand(sp.diff(f(sp.pi), x).subs(x, 1) * (x - 1) + f(sp.pi).subs(x, 1))
    V.check(3, "tangent pi, x=1", tan, E(V.ans(3, "2")))
    disc = sp.discriminant(sp.diff(f(th), x), x)
    sols = [s for s in sp.solveset(disc, th, sp.Interval.open(0, sp.pi))]
    V.check(3, "theta set", sols, [sp.sympify(s) for s in V.ans(3, "3").split(",")])
    T = sp.pi / 3
    fT = f(T); tl = fT.subs(x, 0) + sp.diff(fT, x).subs(x, 0) * x
    xs = sp.solve(sp.expand(fT - tl), x)
    lo, hi = min(xs, key=lambda r: float(r)), max(xs, key=lambda r: float(r))
    V.check(3, "area", sp.integrate(sp.simplify(fT - tl), (x, lo, hi)), V.ans(3, "4"))


if __name__ == "__main__":
    V.run(globals())
