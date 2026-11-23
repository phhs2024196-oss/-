#!/usr/bin/env python3
"""서경대 2026학년도 수시 논술 A형 수학 5~8번 — sympy 검산(기대값은 YAML 정답에서 읽음)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SKU-G26A", "items/SKU/PAST")
p_, q_, c_ = sp.symbols("p q c")


def E(s):
    return S(s, locals={"x": x})


def p05():
    f = x**2 + p_ * x + q_
    # 극한이 존재하려면 분모 0 → 분자 0: f(2)=3, f(3)=f(2)
    sol = sp.solve([f.subs(x, 2) - 3, f.subs(x, 3) - f.subs(x, 2)], [p_, q_])
    g = f.subs(sol)
    V.check(5, "1 f(x)", g, E(V.ans(5, "1")))
    V.check(5, "2 a", sp.limit((g - 3) / (x**2 - 4), x, 2), V.ans(5, "2"))
    V.check(5, "3 b", sp.limit((g - g.subs(x, x - 1)) / (x - 3), x, 3), V.ans(5, "3"))


def p06():
    # 좌표로 직접 구성: C=(0,0), B=(2,0), AB=AD=y, CD=4, 네 점이 한 원 위
    for yy in (sp.Integer(3), sp.Integer(5), sp.Rational(7, 2)):
        # cos(theta)=-1/yy 이면 A = B + yy*(-cos, sin)  (각 ABC = theta, BC 방향은 (-1,0))
        ct = -1 / yy
        st = sp.sqrt(1 - ct**2)
        A = sp.Matrix([2 - yy * ct, yy * st])
        d1, d2 = sp.symbols("d1 d2", real=True)
        sols = sp.solve([d1**2 + d2**2 - 16, (d1 - A[0])**2 + (d2 - A[1])**2 - yy**2], [d1, d2])
        ok = False
        for s in sols:
            Dm = sp.Matrix(s)
            M = sp.Matrix([[pt[0]**2 + pt[1]**2, pt[0], pt[1], 1] for pt in ([0, 0], [2, 0], list(A), list(Dm))])
            if abs(float(M.det())) < 1e-9:
                ok = True
                break
        V.check(6, f"AB={yy}, cos=-1/AB 에서 네 점이 한 원 위", ok, True)
        V.check(6, "AB cos = YAML", yy * ct, V.ans(6, "1"))


def p07():
    a, b = sp.symbols("a b", positive=True)
    k = sp.symbols("k", positive=True)
    # A,B,C 의 x 좌표
    xA = sp.solve(sp.log(x, 2) - 3, x)[0]
    V.check(7, "1 A 의 x좌표", xA, S(V.ans(7, "1"))[0])
    V.check(7, "1 A 의 y좌표", 3, S(V.ans(7, "1"))[1])
    xB, xC = a**3, b**3   # log_a x = 3 ⇔ x = a^3
    kk = sp.solve(sp.Eq(5 * (xB - xA).subs({a: 2 * k, b: 3 * k}), (xC - xB).subs({a: 2 * k, b: 3 * k})), k)
    kk = [s for s in kk if s.is_real and s > 1]
    val = sp.nsimplify((xC - xB).subs({a: 2 * k, b: 3 * k}).subs(k, kk[0]))
    qq, pp = sp.fraction(val)
    V.check(7, "p,q 서로소", sp.gcd(pp, qq), 1)
    V.check(7, "2 p+q", pp + qq, V.ans(7, "2"))


def p08():
    a, C = sp.symbols("a C")
    fp = x**2 - 3 * a
    asol = sp.solve(fp.subs(x, -1), a)[0]
    V.check(8, "1 a", asol, V.ans(8, "1"))
    F = sp.integrate(fp.subs(a, asol), x) + C
    Cs = sp.solve(F.subs(x, -1), C)[0]
    f = F.subs(C, Cs)
    V.check(8, "2 f(x)", f, E(V.ans(8, "2")))
    roots = sp.solve(f, x)
    lo, hi = min(roots), max(roots)
    area = sp.integrate(sp.Abs(f), (x, lo, hi)) if False else sp.integrate(-f, (x, -1, 2))
    # f 의 부호: (-1,2) 에서 f<0 인지 확인
    ok = all(f.subs(x, t) < 0 for t in [sp.Rational(-9, 10), 0, 1, sp.Rational(19, 10)])
    V.check(8, "(-1,2)에서 f<0", ok, True)
    V.check(8, "3 넓이", area, V.ans(8, "3"))


if __name__ == "__main__":
    V.run(globals())
