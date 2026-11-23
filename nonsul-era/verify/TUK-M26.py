#!/usr/bin/env python3
"""TUK-M26-01~06 (한국공학대 2026 모의논술) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

def E(s):
    return sp.sympify(s, locals={"x": x, "n": n, "t": t})


V = Verifier("TUK-M26", "items/TUK/PAST")


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    a = sp.Symbol("a", positive=True)
    sol = sp.solve(sp.log(2, 3) * sp.log(a, 4) + 1, a)
    V.check(1, "a", sol[0], E(V.ans(1, "1")))


def p02():
    th = sp.Symbol("th", real=True)
    g = sp.cos(th) ** 2 + 3 * sp.sin(th) ** 2 - 3 * sp.cos(th) - 1
    # 수치로 부등식의 해 구간 [alpha,beta] 확인
    import math
    xs = [2 * math.pi * i / 200000 for i in range(200000)]
    gf = sp.lambdify(th, g, "math")
    ok = [z for z in xs if gf(z) >= 0]
    al, be = min(ok), max(ok)
    V.check(2, "alpha,beta 수치", (al, be), (math.pi / 3, 5 * math.pi / 3), 1e-4)
    V.check(2, "sin((a+b)/6)", sp.sin((sp.pi / 3 + 5 * sp.pi / 3) / 6), E(V.ans(2, "1")))
    sol = sp.solveset(g, th, sp.Interval.Ropen(0, 2 * sp.pi))
    V.report(2, sol == sp.FiniteSet(sp.pi / 3, 5 * sp.pi / 3), f"등호 성립점 {sol} (= 부등식 해 구간의 양 끝)")


def p03():
    tt = sp.Symbol("tt", positive=True)
    f = lambda z: -3 * z**2 + 3  # noqa: E731
    yq = f(tt) + (-6 * tt) * (0 - tt)  # y절편
    gt = tt * yq / 2
    V.check(3, "g'(1/2)", sp.diff(gt, tt).subs(tt, sp.Rational(1, 2)), V.ans(3, "1"))


def p04():
    k = sp.Symbol("k")
    kk = sp.solve(sp.integrate(k * t**2 + t, (t, 0, 3)) - 9, k)[0]
    v = kk * t**2 + t
    V.check(4, "a(4)", sp.diff(v, t).subs(t, 4), V.ans(4, "1"))


def p05():
    f = x**3 - 2 * x**2 + 2 * x
    V.report(5, bool(sp.discriminant(sp.diff(f, x), x) < 0), "f' 판별식<0")
    xs = sp.solve(f - x, x)
    # 대칭성: 넓이 = 2*∫|f-x| (f-x>=0 on [0,1])
    V.report(5, bool(sp.minimum(f - x, x, sp.Interval(0, 1)) >= 0), "f-x>=0 on [0,1]")
    V.check(5, "area", 2 * sp.integrate(f - x, (x, 0, 1)), V.ans(5, "1"))
    # 수치 독립 확인: 역함수를 수치로 구해 면적 적분
    import mpmath as mp
    fn = sp.lambdify(x, f, "mpmath")
    inv = lambda yv: mp.findroot(lambda z: fn(z) - yv, yv)  # noqa: E731
    area = mp.quad(lambda z: abs(fn(z) - inv(z)), [0, 1])
    V.check(5, "area 수치(역함수)", sp.Float(area, 15), V.ans(5, "1"), 1e-8)


def p06():
    a, b = sp.symbols("a b")
    f = x**3 + 2 * a * x + b
    sol = sp.solve([sp.diff(f, x).subs(x, 1) - sp.Rational(5, 2), sp.integrate(f, (x, 0, 2)) - 3], [a, b], dict=True)[0]
    f = f.subs(sol)
    g = sp.integrate(f.subs(x, t), (t, 0, x))
    crit = sp.solve(sp.diff(g, x), x)
    mx, mn = [], []
    for c in crit:
        s2 = sp.diff(g, x, 2).subs(x, c)
        (mx if s2 < 0 else mn).append(sp.simplify(g.subs(x, c)))
    pr = parts(6, "1")
    V.check(6, "극댓값", mx[0], pr["max"])
    V.check(6, "극솟값(두 곳 동일)", mn, [sp.sympify(pr["min"])] * 2)


if __name__ == "__main__":
    V.run(globals())
