#!/usr/bin/env python3
"""TUK-M25-01~06 (한국공학대 2025 모의논술) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

def E(s):
    return sp.sympify(s, locals={"x": x, "n": n, "t": t})


V = Verifier("TUK-M25", "items/TUK/PAST")


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    a, b = sp.symbols("a b", positive=True)
    f = lambda xx: sp.log(2 * sp.sin(xx) + a, 2) + b  # noqa: E731
    sol = sp.solve([f(0) - 4, f(sp.pi / 2) - 5], [a, b], dict=True)[0]
    V.check(1, "f(pi/6)", sp.simplify(f(sp.pi / 6).subs(sol)), E(V.ans(1, "1")))
    # 수치: 구간에서 최대·최소 확인
    g = sp.lambdify(x, f(x).subs(sol), "math")
    vals = [g(float(sp.pi) * i / 2000) for i in range(0, 2000)]
    V.check(1, "max/min 수치", (max(vals), min(vals)), (5, 4), 1e-5)


def p02():
    S = lambda m: sp.Rational(m, 3 * m + 1)  # noqa: E731
    a = lambda m: S(1) if m == 1 else S(m) - S(m - 1)  # noqa: E731
    V.check(2, "sum", sum((1 - 16 * a(k) ** 2) / a(k) for k in range(1, 6)), V.ans(2, "1"))


def p03():
    f = lambda z: 1 + sp.sin(3 * z)  # noqa: E731
    expr = 2 * f(x - sp.pi / 2) * f(x + sp.pi / 2) - f(x - sp.pi)
    cand = sp.solveset(sp.simplify(expr), x, sp.Interval(0, sp.pi))
    V.check(3, "solutions", list(cand), [E(s) for s in V.ans(3, "1").split(",")])


def p04():
    A = lambda m: sp.integrate(-sp.Rational(1, m) * (x - m + 1) * (x - m), (x, m - 1, m))  # noqa: E731
    V.check(4, "sum 1/A_n", sum(1 / A(m) for m in range(1, 11)), V.ans(4, "1"))


def p05():
    g1, g1p = 4, 2  # 접선 y=2x+2
    G = sp.Function("G")
    ff = (x**2 + 1) * G(x)
    fp1 = sp.diff(ff, x).subs(sp.Derivative(G(x), x), g1p).subs(G(x), g1).subs(x, 1)
    pr = parts(5, "1")
    V.check(5, "f'(1)", fp1, pr["fprime"])
    V.check(5, "limit = f(1)", (x**2 + 1).subs(x, 1) * g1, pr["극한값"])
    # 구체 예: g(x)=2x+2 → 수치로 극한 확인
    gg = 2 * x + 2
    F = sp.integrate(((t**2 + 1) * (2 * t + 2)), (t, 1, x))
    V.check(5, "limit(예 g=2x+2)", sp.limit(F / (x - 1), x, 1), pr["극한값"])
    V.check(5, "f'(1)(예)", sp.diff((x**2 + 1) * gg, x).subs(x, 1), pr["fprime"])


def p06():
    a, b = sp.symbols("a b")
    sol = sp.solve([a * b + 2, a + b - 1], [a, b], dict=True)
    ders = {sp.simplify(sp.diff(s[a] * s[b] * (x - 1) ** 2, x)) for s in sol}
    V.check(6, "(fg)'", list(ders)[0], E(V.ans(6, "1")))
    V.check(6, "해 개수/일치", len(ders), 1)


if __name__ == "__main__":
    V.run(globals())
