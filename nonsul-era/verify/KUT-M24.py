#!/usr/bin/env python3
"""KUT-M24-01~10 (한국기술교육대 2024학년도 온라인 모의논술 수리논술) — sympy 검산. 기대값은 YAML 에서 V.ans() 로 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KUT-M24", "items/KUT/PAST")
a, b, c, m, r, h = sp.symbols("a b c m r h", real=True)


def p01():
    sol = sp.solve([2**(-2 - a) + b, sp.Abs(b) - 4], [a, b], dict=True)  # (가),(나)
    sol = [s for s in sol if s[b] < 0]
    V.check(1, "a+b", (sol[0][a] + sol[0][b]), V.ans(1, "1"))


def p02():
    ks = [k for k in range(1, 31)
          if sp.simplify(sp.log(sp.root(72, 3), 3) + sp.log(sp.root(3**k, 6) / 2, 3)).is_integer is True
          or sp.nsimplify(sp.N(sp.log(sp.root(72, 3), 3) + sp.log(sp.root(3**k, 6) / 2, 3), 40), rational=True).is_integer]
    V.check(2, "합", sum(ks), V.ans(2, "1"))


def p03():
    av = sp.solve(sp.sin(sp.pi / 3)**2 + a * sp.cos(sp.pi / 3) - 1, a)[0]
    roots = sp.solveset(sp.sin(x)**2 + av * sp.cos(x) - 1, x, sp.Interval.Ropen(0, 2 * sp.pi))
    assert len(roots) == 4
    others = [r_ for r_ in roots if sp.simplify(r_ - sp.pi / 3) != 0]
    V.check(3, "a+α+β+γ", av + sum(others), V.ans(3, "1"))


def p04():
    A, B = sp.symbols("A B")
    f = x**3 + 2 * x**2 + A * x + B
    assert sp.limit((f - x**3) / (4 * x**2 + 1), x, sp.oo) == sp.Rational(1, 2)
    sol = sp.solve([f.subs(x, 0), sp.limit(sp.cancel((f / (x**2 + 4 * x)).subs(B, 0)), x, 0) - 2], [A, B], dict=True)[0]
    ff = f.subs(sol)
    V.check(4, "f'(1)", sp.limit((ff - ff.subs(x, 1)) / (x - 1), x, 1), V.ans(4, "1"))


def p05():
    f = a * x**2 + b * x + 7
    sol = sp.solve([f.subs(x, -3) - f.subs(x, -1),
                    (f.subs(x, 0) - f.subs(x, -2)) / 2 - sp.diff(f, x).subs(x, -a / 2)], [a, b], dict=True)
    sol = [s for s in sol if s[a] != 0][0]
    V.check(5, "f'(1)", sp.diff(f, x).subs(x, 1).subs(sol), V.ans(5, "1"))


def p06():
    f1 = -x + 3
    f2 = (a * x + b) / (x + 1)
    sol = sp.solve([f2.subs(x, 0 + 2) * 0 + f2.subs(x, 2) - f1.subs(x, 0), f2.subs(x, 1) - f1.subs(x, 1)], [a, b], dict=True)[0]
    V.check(6, "f(7/2)=f(3/2)", f2.subs(sol).subs(x, sp.Rational(3, 2)), V.ans(6, "1"))


def p07():
    A = sp.symbols("A", real=True)
    f = x**3 + A * x**2 - 3 * A * x + 1
    D = sp.discriminant(sp.diff(f, x), x)
    ints = [k for k in range(-60, 61) if D.subs(A, k) <= 0]
    V.check(7, "정수 a 합", sum(ints), V.ans(7, "1"))
    # 수치 확인: 단조증가
    ok = all(all(f.subs(A, k).subs(x, i + 1) > f.subs(A, k).subs(x, i) for i in [sp.Rational(j, 2) for j in range(-40, 40)]) for k in ints)
    V.report(7, ok, "구한 a 에서 f 단조증가 표본 확인")


def p08():
    f = sp.Function("f")
    A, B, C = sp.symbols("A B C")
    fx = A * x**2 + B * x + C
    F = sp.integrate(fx, x)  # 상수항은 임의
    Fx = F + sp.Symbol("K")
    eq = sp.Poly(sp.expand(x * fx - Fx - x**3 + 2 * x**2), x).all_coeffs()
    sol = sp.solve(eq + [fx.subs(x, 0) - 1], [A, B, C, sp.Symbol("K")], dict=True)
    # 3차 항 계수: f 는 이차(미분하면 3x-4)
    f1 = sp.integrate(3 * x - 4, x) + 1
    Fa = sp.integrate(f1, x)
    assert sp.simplify(x * f1 - (Fa + x**3 - 2 * x**2) + 0).as_poly(x).degree() <= 1  # 상수 K 는 허용
    val = sp.limit(sp.integrate(f1.subs(x, sp.Symbol("s")), (sp.Symbol("s"), 2 - h, 2 + h)) / h, h, 0)
    V.check(8, "극한", val, V.ans(8, "1"))


def p09():
    rr = sp.root(6, 4)
    S = lambda n: sum(rr**i for i in range(n))
    assert sp.simplify(S(8) - 7 * S(4)) == 0
    V.check(9, "1-1 a13", rr**12, V.ans(9, "1-1"))
    ms = [k for k in range(1, 51) if sp.nsimplify(sp.N(S(2 * k) / S(k), 50), rational=True).is_integer]
    V.check(9, "1-2 합", sum(ms), V.ans(9, "1-2"))


def p10():
    f = lambda z: z**3 - 3 * z**2 - 24 * z
    t = sp.symbols("t", real=True)
    left = sp.diff(f(x), x).subs(x, 0)
    right = sp.diff(f(x), x).subs(x, 2 * t)
    ts = [s for s in sp.solve(left - right, t) if s != 0]
    V.check(10, "1-1 t", ts[0], V.ans(10, "1-1"))
    g1 = sp.expand(f(x - 1) + f(2))
    g2 = sp.expand(f(x + 1))
    assert sp.simplify(g1.subs(x, 1) - g2.subs(x, 1)) == 0
    assert sp.simplify(sp.diff(g1, x).subs(x, 1) - sp.diff(g2, x).subs(x, 1)) == 0
    ks = [k for k in range(-200, 201)
          if len({r_ for r_ in sp.Poly(g1 - k, x).real_roots() if r_ < 1}) + len({r_ for r_ in sp.Poly(g2 - k, x).real_roots() if r_ >= 1}) == 3]
    V.check(10, "1-2 개수", len(ks), V.ans(10, "1-2"))


if __name__ == "__main__":
    V.run(globals())
