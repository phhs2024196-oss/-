#!/usr/bin/env python3
"""TUK-M27-01~06 (한국공학대 2027 모의논술) — sympy 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F403

def E(s):
    return sp.sympify(s, locals={"x": x, "n": n, "t": t})


V = Verifier("TUK-M27", "items/TUK/PAST")


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def p01():
    g = 2 * sp.sin(x) ** 2 + sp.cos(sp.pi / 2 + x) - 1
    ss = sp.Symbol("ss")
    cands = set()
    for sv in sp.solve(2 * ss**2 - ss - 1, ss):          # sin x = sv
        a0 = sp.asin(sv)
        for c in (a0, sp.pi - a0, a0 + 2 * sp.pi, 3 * sp.pi - a0):
            if 0 <= float(c) < 2 * float(sp.pi):
                cands.add(sp.simplify(c))
    roots = [c for c in cands if abs(float(g.subs(x, c))) < 1e-12]
    V.check(1, "실근의 합", sum(roots), E(V.ans(1, "1")))
    V.report(1, len(roots) == 3 and len(cands) == 3, f"실근 {sorted(roots, key=float)}")


def p02():
    k = sp.Symbol("k", positive=True)
    PA = 2 ** (k - 1); PB = 4**k + 1; AB = PB - PA
    sol = [s for s in sp.solve(sp.Eq(4 * PA, AB), k)]
    V.check(2, "k", sol[0], V.ans(2, "1"))
    V.report(2, len(sol) == 1, f"양수해 {sol}")


def p03():
    a, b = sp.symbols("a b", positive=True)
    sol = sp.solve([sp.log(a * b, 3) - 1, sp.log(b / a, 3) - 3], [a, b], dict=True)[0]
    pr = parts(3, "1")
    V.check(3, "a", sol[a], pr["a"])
    V.check(3, "b", sol[b], pr["b"])


def p04():
    a = sp.Symbol("a")
    aa = sp.solve(sp.integrate(t**2 + a * t, (t, 0, 3)), a)[0]
    v = t**2 + aa * t
    zs = [z for z in sp.solve(v, t) if 0 < z < 3]
    pts = [0] + zs + [3]
    d = sum(sp.Abs(sp.integrate(v, (t, p, q))) for p, q in zip(pts, pts[1:]))
    V.check(4, "이동 거리", d, V.ans(4, "1"))


def p05():
    import mpmath as mp
    f = x**3 + x + 1
    fn = sp.lambdify(x, f, "mpmath")
    inv = lambda yv: mp.findroot(lambda z: fn(z) - yv, 0.5)  # noqa: E731
    val = mp.quad(inv, [1, 3])
    V.check(5, "∫g (수치, 역함수)", sp.Float(val, 15), V.ans(5, "1"), 1e-8)
    V.check(5, "3*1-∫f", 3 - sp.integrate(f, (x, 0, 1)), V.ans(5, "1"))


def p06():
    tt = sp.Symbol("tt", positive=True)
    f = lambda z: z**2 - 4 * z + 4  # noqa: E731
    xp = sp.solve(f(tt) + sp.diff(f(x), x).subs(x, tt) * (x - tt), x)[0]
    yq = f(tt) + sp.diff(f(x), x).subs(x, tt) * (0 - tt)
    g = sp.simplify(xp * yq / 2)
    V.check(6, "g(t)", sp.expand(g), sp.expand(sp.Rational(1, 4) * (-tt**3 - 2 * tt**2 + 4 * tt + 8)))
    cs = [c for c in sp.solve(sp.diff(g, tt), tt) if 0 < c < 2]
    V.check(6, "최댓값", max(g.subs(tt, c) for c in cs), V.ans(6, "1"))
    V.report(6, len(cs) == 1 and sp.diff(g, tt, 2).subs(tt, cs[0]) < 0, "극대 확인")


if __name__ == "__main__":
    V.run(globals())
