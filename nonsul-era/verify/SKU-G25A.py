#!/usr/bin/env python3
"""서경대 2025학년도 수시 논술 A형 수학 5~8번 — sympy 검산(기대값은 YAML 정답에서 읽음)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SKU-G25A", "items/SKU/PAST")
a, r = sp.symbols("a r", positive=True)


def _num(s):
    return s.replace("\\pi", "*pi")


def p05():
    # B=(0,0), C=(1,0), A=(0,2), M=(0,1)  (BC=1, AB=2)
    A, M, C = sp.Matrix([0, 2]), sp.Matrix([0, 1]), sp.Matrix([1, 0])
    u, v = A - C, M - C
    cos = (u.dot(v)) / (sp.sqrt(u.dot(u)) * sp.sqrt(v.dot(v)))
    sin2 = sp.nsimplify(sp.simplify(1 - cos**2))
    q, p = sp.fraction(sin2)
    V.check(5, "sin^2=q/p, gcd=1", sp.gcd(p, q), 1)
    V.check(5, "p-q", p - q, V.ans(5, "1"))


def p06():
    r3 = r**2
    O1O3 = 1 + 2 * r + r3
    # O1O3^2-(r3-r1)^2 : O1 에서 O3H 에 내린 수선 O1I = GH
    GH2 = sp.expand(O1O3**2 - (r3 - 1)**2)
    V.check(6, "O1I^2=4r(r+1)^2", GH2, 4 * r * (r + 1)**2)
    # 독립 확인: 접점 간 거리 = G~(C2 접점) + (C2 접점)~H = 2sqrt(r1 r2)+2sqrt(r2 r3)
    V.check(6, "GH 이웃 접선 합", sp.simplify(GH2 - (2 * sp.sqrt(r) + 2 * sp.sqrt(r * r3))**2), 0)
    sol = [s for s in sp.solve(4 * r * (r + 1)**2 - 400, r) if s.is_real and s > 1]
    V.check(6, "r", sol[0], 4)
    V.check(6, "넓이합", sp.pi * (sol[0]**2 + (sol[0]**2)**2), _num(V.ans(6, "1")))


def p07():
    a_ = sp.symbols("a_", positive=True)
    tot = sp.integrate(-x**2 + 3 * x, (x, 0, 3))
    S1 = sp.integrate(-x**2 + 3 * x - a_ * x, (x, 0, 3 - a_))
    S2 = tot - S1
    sols = [s for s in sp.solve(sp.Eq(19 * S1, 8 * S2), a_) if s.is_real and 0 < s < 3]
    V.check(7, "a", sols[0], V.ans(7, "1"))
    V.check(7, "S1+S2", tot, sp.Rational(9, 2))


def p08():
    k_ = sp.symbols("k_")
    f = x**4 + x**3 - 2 * x**2 - 8 * x
    g = x**3 + 2 * x**2 + 8 * x
    good = []
    for kk in range(-60, 60):
        cnt = sum(1 for xx in range(-50, 51) if f.subs(x, xx) <= g.subs(x, xx) + kk)
        if cnt == 3:
            good.append(kk)
    V.check(8, "합", sum(good), V.ans(8, "1"))
    V.check(8, "k 목록", good, [-3, -2, -1])


if __name__ == "__main__":
    V.run(globals())
