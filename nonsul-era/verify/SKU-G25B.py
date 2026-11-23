#!/usr/bin/env python3
"""서경대 2025학년도 수시 논술 B형 수학 5~8번 — sympy 검산(기대값은 YAML 정답에서 읽음)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SKU-G25B", "items/SKU/PAST")


def p05():
    # 직선 기울기 -3, 중점 x=3: A=(p,3^p), B=(6-p,3^(8-p)+6)
    p = sp.symbols("p", real=True)
    cond = sp.Eq((3**(8 - p) + 6 - 3**p), -3 * (6 - p - p))
    # 단조성으로 해가 유일함을 수치 확인 후 해 대입
    sol = [s for s in [0, 1, 2, 3, 4, 5, 6, 7, 8] if sp.simplify(cond.lhs.subs(p, s) - cond.rhs.subs(p, s)) == 0]
    V.check(5, "p(유일 정수해)", sol, [4])
    g = lambda s: sp.diff(3**(8 - p) + 6 - 3**p + 3 * (6 - 2 * p), p).subs(p, s)
    V.proof(5, "좌변-우변 증감: 도함수 항상 음수(격자 검사)", [s for s in sp.Range(-20, 21) if g(s) >= 0])
    pv = sol[0]
    A = (pv, 3**pv); B = (6 - pv, 3**(8 - pv) + 6)
    V.check(5, "기울기", (B[1] - A[1]) / (B[0] - A[0]), -3)
    V.check(5, "a", (A[1] + B[1]) / 2, V.ans(5, "1"))


def p06():
    # 점화 a_{n+3}=15-a_n, a1+a2+a3=10 : (a1,a2,a3) 임의 값으로 확인
    for (a1, a2) in [(1, 2), (-3, 7), (sp.Rational(1, 2), 4)]:
        a = {1: a1, 2: a2, 3: 10 - a1 - a2}
        for n_ in range(1, 61):
            a[n_ + 3] = 15 - a[n_]
        V.check(6, f"합 (a1,a2)=({a1},{a2})", sum(a[i] for i in range(1, 64)), V.ans(6, "1"))


def p07():
    k_ = sp.symbols("k_")
    f = -x**3 + 3 * x**2 + 1
    # 곡선-직선 = f(x)-k
    V.check(7, "정리", sp.expand((-x**3 + 3 * x**2 - 3 * x + 1) - (-3 * x)), f)
    crit = sp.solve(sp.diff(f, x), x)
    vals = sorted(f.subs(x, c) for c in crit)
    V.check(7, "극값(극소,극대)", vals, [1, 5])
    good = []
    for kk in range(-5, 12):
        distinct = len(set(sp.Poly(f - kk, x).real_roots()))
        if distinct == 3:
            good.append(kk)
    V.check(7, "합", sum(good), V.ans(7, "1"))


def p08():
    f = x**3 / 4
    gp = sp.cbrt(4 * x)          # x>=0
    gn = -sp.cbrt(-4 * x)        # x<0 (실수 세제곱근)
    pts = sorted(sp.solve(sp.Eq(f, x), x))
    V.check(8, "교점", pts, [-2, 0, 2])
    area = sp.integrate(f - gn, (x, -2, 0)) + sp.integrate(gp - f, (x, 0, 2))
    V.check(8, "g 직접 적분", sp.simplify(area), V.ans(8, "1"))
    V.check(8, "대칭 이용 4배", 4 * sp.integrate(x - x**3 / 4, (x, 0, 2)), V.ans(8, "1"))


if __name__ == "__main__":
    V.run(globals())
