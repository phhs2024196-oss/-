#!/usr/bin/env python3
"""HSN-M25-01~04 (2025학년도 모의논술고사) sympy 검산 — 기대값은 YAML(V.ans)에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("HSN-M25", "items/HSN/PAST")
a, N, th = sp.symbols("a N theta")


def p01():
    # (a^(1/6))^(3/2) = a^(1/4) 가 자연수인 100 이하 자연수 a
    assert sp.simplify(sp.powsimp(sp.Pow(sp.Pow(a, sp.Rational(1, 6)), sp.Rational(3, 2)), force=True) - a ** sp.Rational(1, 4)) == 0
    sols = {v for v in range(1, 101) if sp.integer_nthroot(v, 4)[1]}
    V.check(1, "1 해집합", sols, V.ans(1, "1"))


def p02():
    lhs = sp.summation(k**2, (k, 1, N))
    rhs = sp.summation(3 * k, (k, 1, N))
    roots = [r for r in sp.solve(sp.Eq(lhs, rhs), N) if r.is_integer and r > 0]
    V.check(2, "1 N", roots[0], V.ans(2, "1"))
    V.report(2, len(roots) == 1, f"자연수 해 유일 {roots}")


def p03():
    lhs = sp.sin(th) / (1 - sp.sin(th)) - sp.sin(th) / (1 + sp.sin(th))
    V.report(3, sp.simplify(lhs - 2 * sp.tan(th) ** 2) == 0, "좌변 = 2 tan^2 θ 항등")
    cand = sp.solveset(sp.tan(th) ** 2 - 1, th, sp.Interval.open(sp.pi, 3 * sp.pi / 2))
    sols = [c for c in cand if abs(sp.N(lhs.subs(th, c) - 2)) < 1e-10]
    V.report(3, len(sols) == 1, f"구간 내 해 개수 {len(sols)}")
    V.check(3, "1 θ", sols[0], V.ans(3, "1"))


def p04():
    f1 = -3 * a * 2 + a**2          # x→2- 좌극한
    f2 = 2 * 2 - a                  # f(2)
    rts = sp.solve(sp.Eq(f1, f2), a)
    V.report(4, all(r.is_real for r in rts) and len(rts) == 2, f"서로 다른 두 실근 {rts}")
    V.check(4, "1 합", sum(rts), V.ans(4, "1"))


if __name__ == "__main__":
    V.run(globals())
