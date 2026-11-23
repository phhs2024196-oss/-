#!/usr/bin/env python3
"""TUK-G23A-01~03 (한국공학대 2023 오전) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

V = Verifier("TUK-G23A", "items/TUK/PAST")
s = sp.Symbol("s", real=True)


def kv(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    f = 3 * sp.sin(2 * x) - 1
    sol = [r for r in sp.solveset(sp.Eq(f, 2), x, sp.Interval(0, sp.pi))]
    V.check(1, "1-1 f(x)=2", sol[0], V.ans(1, "1")) if len(sol) == 1 else V.report(1, False, f"1-1 해 {sol}")
    g = lambda u: 2 ** (1 - u) + 3
    fmin, fmax = -4, 2  # 3 sin2x-1 on [0,pi]
    V.check(1, "1-2 최댓값 g(fmin)", g(sp.Integer(fmin)), kv(1, "2")["max"])
    V.check(1, "1-2 최솟값 g(fmax)", g(sp.Integer(fmax)), kv(1, "2")["min"])
    roots = sorted(sp.solveset(sp.Eq(f, -3), x, sp.Interval(0, sp.pi)), key=lambda r: float(r))
    V.check(1, "1-3 a+b", sum(roots) if len(roots) == 2 else None, V.ans(1, "3"))
    h = f - (sp.sin(2 * x) - sp.sin(2 * x - sp.pi / 2) ** 2)  # f - sin2x + sin^2(..) >= k  -> min of h
    import numpy as np
    xs = np.linspace(0, 2 * np.pi, 200001)
    hv = sp.lambdify(x, h, "numpy")(xs)
    kmax = float(hv.min())
    kb = V.ans(1, "4").replace("k <=", "").strip()
    V.report(1, abs(kmax - float(S(kb))) < 1e-9, f"1-4 k 상한 = min h = {kmax} / YAML {kb}")


def p02():
    a = [None, 1, 2]
    for i in range(1, 4): a.append(sp.Rational(a[i + 1] ** 2, a[i]))
    V.check(2, "2-1 a5", a[5], V.ans(2, "1"))
    b = [None, 1, 3]
    for i in range(1, 5): b.append(2 * b[i + 1] - b[i])
    V.check(2, "2-2 Σ(1+b)^2", sum((1 + b[i]) ** 2 for i in range(1, 7)), V.ans(2, "2"))
    c = [None, sp.Integer(6), sp.Integer(4)]
    for i in range(1, 7):
        c.append(c[i + 1] ** 2 / c[i] if c[i] < 5 else 2 * c[i + 1] - c[i])
    V.check(2, "2-3 c8", c[8], V.ans(2, "3"))
    f = x**3 / 3 - x**2 / 2 - 20 * x + 3
    cr = sp.solve(sp.diff(f, x), x)
    pmin = [r for r in cr if sp.diff(f, x, 2).subs(x, r) > 0][0]
    qmax = [r for r in cr if sp.diff(f, x, 2).subs(x, r) < 0][0]
    d1, d = sp.symbols("d1 d")
    sol = sp.solve([d1 + 3 * d - pmin, d1 + 6 * d - qmax], [d1, d])
    V.check(2, "2-4 d_n", sol[d1] + (n - 1) * sol[d], sp.sympify(V.ans(2, "4"), locals={"n": n}))


def p03():
    f = x**3 + 3 * x**2 - 9 * x + 2
    cr = sp.solve(sp.diff(f, x), x)
    V.check(3, "3-1 극솟값", [f.subs(x, r) for r in cr if sp.diff(f, x, 2).subs(x, r) > 0][0], V.ans(3, "1"))
    g = (x**4 - x**2 + 1) * f   # d/dx ∫_1^x g = g(x)
    gt = sp.Function("G")
    gg = sp.diff(sp.integrate(g.subs(x, t), (t, 1, x)), x)
    V.check(3, "3-2 g'(1)", sp.diff(gg, x).subs(x, 1), V.ans(3, "2"))
    m = sp.Symbol("m")
    p = f + m * x**2 + 10 * x - 2
    ok = [mm for mm in range(-20, 21) if sp.discriminant(sp.diff(p, x).subs(m, mm), x) <= 0]
    V.check(3, "3-3 정수 m (p'≥0 ⇔ 판별식≤0)", sp.FiniteSet(*ok), sp.FiniteSet(*[S(v) for v in V.ans(3, "3").replace(" ", "").split(",")]))
    h = f - 4 * x**2 + 10 * x - 2
    sols = sorted(sp.solve(h - x, x))
    area = 2 * sp.integrate(x - h, (x, sols[0], sols[-1]))  # h 와 역함수는 y=x 에 대칭, 접선이 y=x 인 점에서 만남
    # 수치 독립 확인: 역함수 k 를 수치로 구해 ∫|h-k|
    import mpmath as mp
    mp.mp.dps = 25
    hf = sp.lambdify(x, h, "mpmath")
    kf = lambda y_: mp.findroot(lambda u: hf(u) - y_, y_)
    num = mp.quad(lambda u: abs(hf(u) - kf(u)), [0, 1])
    V.report(3, abs(num - mp.mpf(str(sp.N(area, 25)))) < 1e-9 and abs(sp.N(area - S(V.ans(3, "4")))) < 1e-12, f"3-4 넓이 {area} (수치 {num})")


if __name__ == "__main__": V.run(globals())
