#!/usr/bin/env python3
"""SKU 2023학년도 수시 논술 수학 1~6번 검산 — 기대값은 YAML(V.ans)에서 읽는다."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SKU-G23", "items/SKU/PAST")


def p01():
    cnt = [xx for xx in range(-3, 100) if xx != 0 and xx + 2 > 0 and abs(xx) * (xx + 2) <= 5**4]  # log_5|x|(x+2)<=4
    V.check(1, "1 정수 개수", len(cnt), V.ans(1, "1"))


def p02():
    sols = set()
    for kk in range(-3, 10):
        for c in (sp.pi / 6, 5 * sp.pi / 6, 7 * sp.pi / 6, 11 * sp.pi / 6):
            v = (c + 2 * sp.pi * kk) / 3
            if 0 <= v < 2 * sp.pi:
                sols.add(sp.simplify(v))
    assert all(sp.simplify(abs(sp.cos(3 * v)) - sp.sqrt(3) / 2) == 0 for v in sols)
    a = len(sols); b = sp.simplify(sum(sols) / sp.pi)
    V.check(2, "2 b/a", sp.Rational(b, a), V.ans(2, "1"))


def p03():
    a, b = sp.symbols("a b")
    def lhs(nn): return sum(k_ * (nn - k_ + 1) for k_ in range(2, nn))
    # a_k = k(n-k+1): a_1 = n, a_n = n 확인 후 합 = n(n+a)(n+b)/6 의 a,b 를 n=3..8 에서 풀기
    ok = all(k_ * (nn - k_ + 1) == nn for nn in range(2, 9) for k_ in (1, nn))
    sol = sp.solve([sp.Eq(lhs(nn) * 6, nn * (nn + a) * (nn + b)) for nn in (3, 4)], [a, b], dict=True)
    ab = {sp.Rational(s[a] + s[b]) for s in sol}
    chk = all(lhs(nn) * 6 == nn * (nn - 2) * (nn + 5) for nn in range(3, 15))
    V.check(3, "3 a+b", ab.pop() if (ok and chk and len(ab) == 1) else None, V.ans(3, "1"))


def p04():
    a, b, kk = sp.symbols("a b kk")
    f = lambda v: a * v + b
    sol = sp.solve([sp.Eq(f(1), 2), sp.Eq(f(-1), 4)], [a, b], dict=True)[0]
    F = f(x).subs(sol)
    k_ = sp.limit((F**2 - 4) / (sp.sqrt(F.subs(x, -x)) - 2), x, 1)
    V.check(4, "4 k/(a+b)", sp.simplify(k_ / (sol[a] + sol[b])), V.ans(4, "1"))


def p05():
    # f(x)=c x 꼴 외에도 임의 다항식에서 상계: |f(4/3)|<=12, f=±9x 가 끝값 실현
    c = sp.symbols("c")
    F = 9 * x
    ok = F.subs(x, 0) == 0 and abs(sp.diff(F, x)) <= 9
    M = F.subs(x, sp.Rational(4, 3)); m = (-F).subs(x, sp.Rational(4, 3))
    V.check(5, "5 M^2-mM+m^2", M**2 - m * M + m**2 if ok else None, V.ans(5, "1"))


def p06():
    f = x**2 - 4 * x - 1
    g = (-f).subs(x, x + 1) + 3        # x축 대칭 후 x축 방향 -1, y축 방향 +3
    rts = sp.solve(sp.Eq(f, g), x)
    lo, hi = min(rts), max(rts)
    V.check(6, "6 넓이", sp.integrate(g - f, (x, lo, hi)), V.ans(6, "1"))


if __name__ == "__main__": V.run(globals())
