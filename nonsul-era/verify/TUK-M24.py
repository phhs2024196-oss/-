#!/usr/bin/env python3
"""TUK-M24-01~03 (한국공학대 2024 모의논술) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

def E(s):
    return sp.sympify(s, locals={"x": x, "n": n, "t": t})


V = Verifier("TUK-M24", "items/TUK/PAST")


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    # 1-1: 2^x=10 -> x=log2(10), 5^(3y)=10 -> y=log5(10)/3
    X = sp.log(10) / sp.log(2); Y = sp.log(10) / sp.log(5) / 3
    pr = parts(1, "1-1")
    V.check(1, "(가) 10^(3/x)", sp.simplify(sp.exp(sp.log(10) * 3 / X)), pr["가"])
    V.check(1, "(나) 10^(1/y)", sp.simplify(sp.exp(sp.log(10) / Y)), pr["나"])
    V.check(1, "(다) 10^(3/x+1/y)", sp.simplify(sp.exp(sp.log(10) * (3 / X + 1 / Y))), pr["다"])
    V.check(1, "(라) 3/x+1/y", sp.simplify(3 / X + 1 / Y), pr["라"])
    V.check(1, "(마) log3(...)", sp.simplify(sp.log(3 / X + 1 / Y) / sp.log(3)), pr["마"])
    a = sp.Symbol("a", positive=True)
    sols = [s for s in sp.solve(2 * a**2 - (7 * a - 3), a) if s > 1]
    V.check(1, "1-2 f(2)", sols[0] ** 2, V.ans(1, "1-2"))
    y = -2 * sp.cos(x) ** 2 + 2 * sp.sin(x) + 1
    lo, hi = extrema(sp.simplify(y.rewrite(sp.sin)), x, 0, 2 * sp.pi)
    pr = parts(1, "1-3")
    V.check(1, "1-3 max", hi, pr["max"])
    V.check(1, "1-3 min", lo, pr["min"])


def p02():
    r = sp.Symbol("r", positive=True)
    rr = sp.solve(sp.Rational(1, 3) * r + sp.Rational(1, 3) * r**2 - 2, r)[0]
    nmin = next(i for i in range(1, 100) if sp.Rational(1, 3) * rr ** (i - 1) > 40)
    V.check(2, "2-1 n min", nmin, V.ans(2, "2-1"))
    a, b, c, d = sp.symbols("a b c d")
    f = a * x**3 + b * x**2 + c * x + d
    # f(0)=0, f'(0)>0, f'(0)*2f'(x)=2x^2+18
    fp = sp.diff(f, x)
    eqs = sp.Poly(fp.subs(x, 0) * 2 * fp - (2 * x**2 + 18), x).all_coeffs() + [f.subs(x, 0)]
    sols = [s for s in sp.solve(eqs, [a, b, c, d], dict=True) if s[c] > 0]
    V.check(2, "2-2 f(x)", sp.expand(f.subs(sols[0])), E(V.ans(2, "2-2")))
    A = sp.Symbol("A", positive=True)
    pa = sp.integrate(3 * t**2 - 6 * t, (t, 0, A)); qa = sp.integrate(4 * t, (t, 0, A))
    a0 = sp.solve(pa - qa, A)[0]
    v1 = 3 * t**2 - 6 * t
    zs = [z for z in sp.solve(v1, t) if 0 < z < a0]  # 속도 부호가 바뀌는 시각
    pts = [0] + zs + [a0]
    s = sum(sp.Abs(sp.integrate(v1, (t, p, q))) for p, q in zip(pts, pts[1:]))
    V.check(2, "2-3 s", s, V.ans(2, "2-3"))


def p03():
    a, b = sp.symbols("a b")
    # 연속: -2a-5=a^2 b-4 / 미분가능: -2 = 2ab
    sols = sp.solve([-2 * a - 5 - (a**2 * b - 4), -2 - 2 * a * b], [a, b], dict=True)
    pr = parts(3, "3-1")
    V.check(3, "3-1 a", sols[0][a], pr["a"])
    V.check(3, "3-1 b", sols[0][b], pr["b"])


if __name__ == "__main__":
    V.run(globals())
