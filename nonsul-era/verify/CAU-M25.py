#!/usr/bin/env python3
"""CAU-M25 (2025학년도 중앙대 모의논술 자연계열) — sympy/전수 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

PREFIX = "CAU-M25"
WIDTH = 2
RESULTS = []
x, y, t, n = sp.symbols("x y t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """주사위 6가지 × 카드 2장 전수 → 점수의 기댓값 (답 146/63)."""
    E = Fr(0)
    for die in range(1, 7):
        cards = [1, 2, 3] if die % 3 == 0 else list(range(2, 9))
        pairs = list(itertools.combinations(cards, 2))
        for a, b in pairs:
            if b**3 - 6 * a * b**2 + 11 * a**2 * b - 6 * a**3 == 0:
                sc = Fr(b, a)
            else:
                sc = Fr(b - a)
            E += Fr(1, 6) * Fr(1, len(pairs)) * sc
    # 대표 오답: (2,8)에서 b=4a 를 몫으로 처리
    report(1, E == Fr(146, 63), f"E(X)={E}")


def p02():
    """(1) d=1..2000 전수로 a_m=0 인 최소 m 이 7인 d (답 7, 18)
    (2) 접선 l_n, l_{n+1} 을 직접 구성해 극한 (답 (e-1)^2/(2e))."""
    good = []
    for d in range(1, 2001):
        a, m = 2, 1
        first = None
        while m <= 7:
            if a == 0:
                first = m
                break
            a = a - 4 if a > 0 else a + d
            m += 1
        if first == 7:
            good.append(d)
    ok1 = good == [7, 18]

    k = sp.symbols("k", positive=True)

    def tangent(k):
        s = sp.symbols("s", positive=True)
        sol = sp.solve(sp.Eq(sp.log(s) + (0 - s) / s, k), s)  # (0,k)를 지나는 접선의 접점
        s0 = sol[0]
        return sp.simplify(1 / s0), k                           # 기울기, y절편
    m1, c1 = tangent(n)
    m2, c2 = tangent(n + 1)
    xi = sp.simplify((c2 - c1) / (m1 - m2))
    yi = sp.simplify(m1 * xi + c1)
    base = sp.simplify((-c1 / m1) - (-c2 / m2))   # 두 x절편 사이 길이(l_n 쪽이 오른쪽)
    assert float((base).subs(n, 3)) > 0 and float(yi.subs(n, 3)) > 0
    S = base * yi / 2
    tdiff = (m1 - m2) / (1 + m1 * m2)
    L = sp.limit(sp.simplify(S * tdiff / n**2), n, sp.oo)
    ok2 = sp.simplify(L - (sp.E - 1) ** 2 / (2 * sp.E)) == 0
    report(2, ok1 and ok2, f"(1) d={good}  (2) 극한={sp.simplify(L)}")


def p03():
    """(1) a_n 을 n=0..8 직접 적분해 (-1)^n/(n+1) 확인 후 합 (답 -506/1013)
    (2) 곡선을 sympy 로 인수분해해 각 분지에서 PA^2+PB^2 최솟값 (답 18)."""
    ok_form = True
    for k in range(0, 9):
        v = sp.integrate((x * sp.log(x) - x) ** k * sp.log(x), (x, 1, sp.E))
        ok_form &= sp.simplify(v - sp.Rational((-1) ** k, k + 1)) == 0
    an = lambda k: Fr((-1) ** k, k + 1)
    tot = sum(an(k) * an(k + 1) for k in range(1, 2025))
    ok1 = ok_form and tot == Fr(-506, 1013)

    poly = y**2 - (x**2 + x + 1) * y + x**3 + x**2
    facs = sp.factor_list(poly, y)[1]
    best = None
    for fac, _ in facs:
        ysol = sp.solve(fac, y)[0]
        F = (x - 1) ** 2 + ysol**2 + (x - 5) ** 2 + ysol**2
        crit = [r for r in sp.solve(sp.diff(F, x), x) if r.is_real]
        vals = [F.subs(x, r) for r in crit]
        mn = min(vals)
        best = mn if best is None else min(best, mn)
    ok2 = best == 18
    report(3, ok1 and ok2, f"(1) 합={tot}  (2) 분지 {[str(f) for f, _ in facs]}, 최솟값={best}")


def p04():
    """(1) 구 S 위 중심 (a,b,c)에서 A/π=147-(a²+b²+c²)의 최솟값+최댓값을 라그랑주로 (답 192)
    (2) 둘레 최댓값=4a 수치 확인, 접선 거리로 b, 직선과 타원 교점으로 넓이 (답 3√5)."""
    a, b, c, lam = sp.symbols("a b c lam", real=True)
    g = (a - 3) ** 2 + (b - 4) ** 2 + (c - 5) ** 2 - 1
    sols = sp.solve([2 * a - lam * 2 * (a - 3), 2 * b - lam * 2 * (b - 4), 2 * c - lam * 2 * (c - 5), g], [a, b, c, lam], dict=True)
    vals = [sp.simplify(147 - (s[a] ** 2 + s[b] ** 2 + s[c] ** 2)) for s in sols]
    # 세 원이 실제로 생기는지(|a|,|b|,|c|<7)
    real_circ = all(abs(float(s[v])) < 7 for s in sols for v in (a, b, c))
    ok1 = sp.simplify(sum(vals) - 192) == 0 and real_circ

    # (가) 둘레 최댓값이 4a 임을 a=√10, b²=6 에서 수치로(점 2개 격자)
    mp.mp.dps = 15
    A_, B2 = mp.sqrt(10), 6
    ok_per = True
    C_ = mp.sqrt(A_**2 - B2)
    pts = [(A_ * mp.cos(2 * mp.pi * i / 360), mp.sqrt(B2) * mp.sin(2 * mp.pi * i / 360)) for i in range(360)]
    dist = lambda P, Q: mp.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
    mx = max(dist(P, Q) + dist(P, (C_, 0)) + dist(Q, (C_, 0)) for P in pts for Q in pts)
    ok_per = (4 * A_ - 1e-2) < mx <= 4 * A_ + 1e-9
    # (나) 기울기 √3 접선 두 개 사이 거리 6 → b²
    bb = sp.symbols("bb", positive=True)
    k = sp.sqrt(10 * 3 + bb)
    bsol = sp.solve(sp.Eq(2 * k / sp.sqrt(1 + 3), 6), bb)
    c_ = sp.sqrt(10 - bsol[0])
    # 교점
    xs = sp.solve(x**2 / 10 + (x + c_) ** 2 / bsol[0] - 1, x)
    P = (xs[0], xs[0] + c_)
    Q = (xs[1], xs[1] + c_)
    F = (c_, 0)
    area = sp.Abs((P[0] - F[0]) * (Q[1] - F[1]) - (Q[0] - F[0]) * (P[1] - F[1])) / 2
    ok2 = ok_per and bsol == [6] and c_ == 2 and sp.simplify(area - 3 * sp.sqrt(5)) == 0
    report(4, ok1 and ok2, f"(1) 합={sp.nsimplify(sum(vals))}  (2) 둘레max≈{mp.nstr(mx, 12)}(4a={mp.nstr(4 * A_, 12)}), b²={bsol}, c={c_}, 넓이={sp.simplify(area)}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
