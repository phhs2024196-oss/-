#!/usr/bin/env python3
"""HSN-M25G-05~08 (2025학년도 논술가이드북 모의문제) sympy 검산 — 기대값은 YAML(V.ans)에서 읽는다."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("HSN-M25G", "items/HSN/PAST")
xx = sp.symbols("xx", positive=True)


def p05():
    # 3^(a+b)=4, 2^(a-b)=7 에서 a, b 를 직접 구해 3^(a^2-b^2) 계산
    s = sp.log(4, 3)   # a+b
    d = sp.log(7, 2)   # a-b
    a_, b_ = (s + d) / 2, (s - d) / 2
    V.check(5, "1 값", sp.N(3 ** (a_**2 - b_**2), 40), V.ans(5, "1"), tol=1e-12)


def p06():
    d = sp.symbols("d")
    dd = sp.solve(sp.Eq(sum(1 + (m - 1) * d for m in range(1, 11)), 100), d)
    V.report(6, len(dd) == 1, f"공차 {dd}")
    V.check(6, "1 a2025", 1 + 2024 * dd[0], V.ans(6, "1"))


def p07():
    L1 = sp.limit((3 * xx - 7) * (2 * xx**2 + 1) / (xx * (xx**2 + 7)), xx, sp.oo)
    L2 = sp.limit(3 * xx * (2 * xx**2 + 1) / (xx * (xx**2 + 3)), xx, sp.oo)
    V.report(7, L1 == L2, f"양끝 극한 일치 {L1}, {L2}")
    V.check(7, "(가)", L1, V.ans(7, "1", "(가)"))
    # f(x)/x -> 6 이므로 f(x)=6x 꼴로 두고 극한 재계산
    f = 6 * xx
    V.check(7, "(나)", sp.limit((4 * xx + 2 * f) / (8 * xx - f), xx, sp.oo), V.ans(7, "1", "(나)"))


def p08():
    f = -x * (x - 1) * (x - 2)
    V.report(8, sp.Poly(f, x).LC() == -1 and sorted(sp.roots(f, x)) == [0, 1, 2], "최고차항 -1, 근 0,1,2")
    area = abs(integral(f, x, 0, 1)) + abs(integral(f, x, 1, 2))
    V.check(8, "1 넓이", area, V.ans(8, "1"))


if __name__ == "__main__":
    V.run(globals())
