#!/usr/bin/env python3
"""KHU-M23 (경희대 2023학년도 모의논술 자연계 논제 I~III) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-M23"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b = sp.symbols("a b", positive=True)
    f = x * (x + a) * (x - b)
    # 부호: (-a,0)에서 양, (0,b)에서 음 — 표본점으로 확인
    sgn_ok = all(f.subs({a: av, b: bv, x: xv}) > 0 for av, bv in [(1, 2), (3, 1)] for xv in [-av / 2])
    sgn_ok &= all(f.subs({a: av, b: bv, x: bv / 2}) < 0 for av, bv in [(1, 2), (3, 1)])
    A = sp.integrate(f, (x, -a, 0))
    B = -sp.integrate(f, (x, 0, b))
    ok1 = sp.simplify(A - a**3 * (a + 2 * b) / 12) == 0 and sp.simplify(B - b**3 * (2 * a + b) / 12) == 0
    # (2) B=2A, t=b/a 로 정리 → 양의 근이 오직 (1,2)에 하나
    eq = sp.expand((B - 2 * A).subs(b, t * a) / a**4)
    poly = sp.Poly(sp.expand(eq * 12), t)
    roots = [r for r in poly.nroots(n=30) if abs(sp.im(r)) < 1e-20 and sp.re(r) > 0]
    ok2 = len(roots) == 1 and 1 < roots[0] < 2
    same = sp.simplify(poly.as_expr() - (t**4 + 2 * t**3 - 4 * t - 2)) == 0 or \
        sp.simplify(poly.as_expr() + (t**4 + 2 * t**3 - 4 * t - 2)) == 0
    report(1, sgn_ok and ok1 and ok2 and same, f"A={sp.factor(A)}, B={sp.factor(B)}; b/a 양근={roots} ∈(1,2) [증명-수치확인]")


def p02():
    p = sp.symbols("p", positive=True)
    ok_all = True
    for pv in [sp.Rational(1, 5), sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(9, 10)]:
        d = sp.sqrt(1 + pv**2)               # 반대각선
        A, C = sp.Matrix([d, 0]), sp.Matrix([-d, 0])
        bx, by = sp.symbols("bx by", real=True)
        sols = sp.solve([(bx - A[0])**2 + by**2 - 4, (bx - C[0])**2 + by**2 - 4 * pv**2], [bx, by], dict=True)
        Bp = [s for s in sols if s[by] > 0][0]
        B = sp.Matrix([Bp[bx], Bp[by]])
        D = -B
        # E: AB와 y축, F: CD와 y축
        sE = sp.solve(A[0] + t * (B[0] - A[0]), t)[0]
        E = A + sE * (B - A)
        a2, b2 = A[0]**2, E[1]**2
        focus = sp.sqrt(sp.simplify(a2 - b2))
        ok1 = sp.simplify(focus - sp.sqrt(1 - pv**4)) == 0 and a2 > b2
        F = C + sp.solve(C[0] + t * (D[0] - C[0]), t)[0] * (D - C)
        ok1 &= sp.simplify(F[1] + E[1]) == 0
        M = (A + B) / 2
        n2 = sp.simplify(M[1]**2 / (1 - M[0]**2 / a2))
        ok2 = sp.simplify(n2 - b2 / (1 + 2 * pv**2)) == 0 and n2 < b2
        ok_all &= bool(ok1 and ok2)
    report(2, ok_all, "p=1/5,1/2,3/4,9/10: 초점 ±√(1-p^4), n^2=b^2/(1+2p^2)<b^2 [증명-수치확인]")


def p03():
    pu, u, dd, A0 = sp.Rational(1, 3), 2, sp.Rational(1, 2), 20
    E = 0
    E_alt = 0
    for path in itertools.product([1, 0], repeat=3):
        prices = [sp.Integer(A0)]
        pr = 1
        for s in path:
            prices.append(prices[-1] * (u if s else dd))
            pr *= pu if s else 1 - pu
        E += pr * (prices[-1] - min(prices))          # 공식 해석: 현재 가격 포함 최저가
        E_alt += pr * (prices[-1] - min(prices[1:]))  # 대안 해석: A1~A3 중 최저가
    ok1 = E == sp.Rational(100, 9)
    # (2)
    p, beta, uu, d, a0 = sp.symbols("p beta u d A0", positive=True)
    V0 = p * (uu * a0 - a0) + (1 - p) * 0
    sol = sp.solve([V0 + beta * a0 - ((uu * a0 - a0) + beta * uu * a0), V0 + beta * a0 - (0 + beta * d * a0)], [p, beta], dict=True)[0]
    ok2 = sp.simplify(sol[p] - (1 - d) / (uu - d)) == 0 and sp.simplify(sol[beta] - (1 - uu) / (uu - d)) == 0
    report(3, ok1 and ok2, f"(1) E={E} (A1~A3 최저 해석이면 {E_alt}); (2) p={sp.simplify(sol[p])}, β={sp.simplify(sol[beta])}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
