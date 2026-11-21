#!/usr/bin/env python3
"""KHU-G22A (경희대 2022학년도 논술 자연계 11.20 오후, KHU-G22A-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-G22A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I (1) lim b_{n+1}/b_n, (2) 정삼각형 단면 입체 부피"""
    s = sp.symbols("s", positive=True)          # 접점 x좌표
    N = sp.symbols("N", positive=True, integer=True)
    # (1) 접선 y = ln s + (x-s)/s 이 (0,-N) 을 지남
    sol = sp.solve(sp.Eq(sp.log(s) + (0 - s) / s, -N), s)
    ok1 = len(sol) == 1
    s0 = sol[0]
    xint = sp.solve(sp.Eq(sp.log(s0) + (x - s0) / s0, 0), x)[0]
    bN = sp.simplify(xint)
    lim = sp.limit(sp.simplify(bN.subs(N, N + 1) / bN), N, sp.oo)
    ok1 = ok1 and sp.simplify(lim - sp.exp(-1)) == 0          # md: 1/e
    # (2) 원점에서 그은 접선: ln s - 1 = 0
    s1 = sp.solve(sp.Eq(sp.log(s) - 1, 0), s)[0]
    slope = 1 / s1
    xx = sp.symbols("xx", positive=True)
    # 구간별 단면 한 변: [0,1] 은 접선 - 0, [1,e] 은 접선 - ln x  (두 경계 확인)
    side1 = slope * xx
    side2 = slope * xx - sp.log(xx)
    ok2 = sp.simplify(side2.subs(xx, s1)) == 0
    # 수치로 영역 경계 재확인: [1,e] 에서 접선 ≥ ln x ≥ 0
    ok2 = ok2 and all(float(side2.subs(xx, 1 + k * (sp.E - 1) / 50)) >= -1e-12 for k in range(51))
    area = lambda L: sp.sqrt(3) / 4 * L**2
    V = sp.integrate(area(side1), (xx, 0, 1)) + sp.integrate(area(side2), (xx, 1, sp.E))
    md = sp.sqrt(3) / 2 * (5 * sp.E / 12 - 1 - 1 / (4 * sp.E))
    ok2 = ok2 and sp.simplify(V - md) == 0
    report(1, ok1 and ok2, f"(1) b_n={bN}, lim={lim}  (2) V={sp.nsimplify(sp.simplify(V))}≈{float(V):.6f}")


def bracket_probs(p, q, r):
    def w(i, j):   # i 가 j 를 이길 확률 (i<j: 상위)
        d = abs(i - j)
        v = {1: p, 2: q, 3: r}[d]
        return v if i < j else 1 - v

    def game(dist_a, dist_b):
        out = {}
        for i, pi in dist_a.items():
            for j, pj in dist_b.items():
                out[i] = out.get(i, 0) + pi * pj * w(i, j)
                out[j] = out.get(j, 0) + pi * pj * w(j, i)
        return out
    one = lambda k: {k: 1}
    b1 = game(game(one(1), one(4)), game(one(2), one(3)))
    b2 = game(one(1), game(one(2), game(one(3), one(4))))
    return sp.expand(b1[1]), sp.expand(b2[1])


def p02():
    """논제 II 대진표 확률"""
    P1, P2 = bracket_probs(sp.Rational(6, 10), sp.Rational(7, 10), sp.Rational(8, 10))
    ok1 = P1 == sp.Rational(512, 1000) and P2 == sp.Rational(648, 1000)    # md: 0.512, 0.648
    p = sp.symbols("p", real=True)
    Q1, Q2 = bracket_probs(p, sp.Rational(5, 6), 1)
    ok2 = sp.expand(Q1 - (6 * p**2 - 5 * p + 5) / 6) == 0 and sp.expand(Q2 - (6 * p**3 - 10 * p**2 + 9 * p + 1) / 6) == 0
    roots = [rt for rt in sp.solve(sp.Eq(Q1, Q2), p) if sp.Rational(1, 2) <= rt < sp.Rational(5, 6)]
    ok2 = ok2 and roots == [sp.Rational(2, 3)]
    report(2, ok1 and ok2, f"(1) P1={P1}, P2={P2}  (2) P1={sp.factor(Q1)}, P2={sp.factor(Q2)}, p={roots}")


def p03():
    """논제 III y=1/x 할선과 삼각형 OPQ"""
    a, b = sp.symbols("a b", positive=True)
    line = lambda X: (1 / b - 1 / a) / (b - a) * (X - a) + 1 / a
    Py = sp.simplify(line(0))
    Qx = sp.solve(sp.Eq(line(x), 0), x)[0]
    AB2 = (a - b)**2 + (1 / a - 1 / b)**2
    PQ2 = Qx**2 + Py**2
    S = sp.simplify(Py * Qx / 2)
    # (1) AB = PQ/2  →  4 AB^2 = PQ^2
    bs = [r for r in sp.solve(sp.Eq(4 * AB2, PQ2), b) if r.is_positive is not False]
    vals = {sp.simplify(S.subs(b, r)) for r in bs}
    ok1 = len(bs) == 2 and vals == {sp.Rational(8, 3)}
    # (2) AB=1: 각 a 에 대해 b 를 수치로 구해 S 계산, a→∞, a→0 경향
    import mpmath as mp
    mp.mp.dps = 60

    def S_of(av, branch):
        av = mp.mpf(av)
        g = lambda bv: mp.sqrt((av - bv)**2 + (1 / av - 1 / bv)**2) - 1
        lo, hi = (av, av + 2) if branch > 0 else (av / 2, av)   # g(a)=-1<0, 반대쪽 끝 g>0
        if branch < 0:
            while g(lo) < 0:
                lo /= 2
        for _ in range(400):
            mid = (lo + hi) / 2
            if (g(mid) < 0) == (branch > 0):
                lo = mid
            else:
                hi = mid
        bv = (lo + hi) / 2
        return float((av + bv)**2 / (2 * av * bv)), float(bv)
    big = [S_of(1e4, +1)[0], S_of(1e4, -1)[0]]
    small = [S_of(1e-3, +1)[0], S_of(1e-3, -1)[0]]
    ok2 = all(abs(v - 2) < 1e-3 for v in big + small)
    # 기호적 확인: b = a(1+u) (u→0) 이면 S = (2+u)^2/(2(1+u)) → 2
    u = sp.symbols("u")
    ok2 = ok2 and sp.limit((2 + u)**2 / (2 * (1 + u)), u, 0) == 2
    report(3, ok1 and ok2, f"(1) b∈{bs}, S={vals}  (2) S(1e4)≈{big}, S(1e-3)≈{small}")


def p04():
    """논제 IV 추적 곡선 r=e^{-αt}"""
    al = sp.symbols("alpha", positive=True)
    N = sp.symbols("N", integer=True, positive=True)
    th = 2 * sp.pi / N
    alpha_expr = (1 - sp.cos(th)) / sp.sin(th)
    f1, g1 = sp.exp(-al * t) * sp.cos(t), sp.exp(-al * t) * sp.sin(t)
    f2, g2 = sp.exp(-al * t) * sp.cos(t + th), sp.exp(-al * t) * sp.sin(t + th)
    cross = sp.diff(f1, t) * (g2 - g1) - sp.diff(g1, t) * (f2 - f1)
    dot = sp.diff(f1, t) * (f2 - f1) + sp.diff(g1, t) * (g2 - g1)
    ok1 = True
    for Nv in (3, 4, 5, 7, 12):
        av = alpha_expr.subs(N, Nv)
        for tv in (0, sp.Rational(1, 3), 2, 5):
            c = sp.N(cross.subs({al: av, N: Nv, t: tv}), 30)
            d = sp.N(dot.subs({al: av, N: Nv, t: tv}), 30)
            ok1 = ok1 and abs(c) < 1e-20 and d > 0
    # 수치 추적 시뮬레이션(n=4): P1 의 극좌표 r 이 e^{-α·각} 과 일치
    import math
    Nv = 4
    av = float(alpha_expr.subs(N, Nv))
    pts = [[math.cos(2 * math.pi * k / Nv), math.sin(2 * math.pi * k / Nv)] for k in range(Nv)]
    dt = 1e-4
    ang_prev, ang_total, dist = 0.0, 0.0, 0.0
    for _ in range(12000):
        new = []
        for k in range(Nv):
            dx = pts[(k + 1) % Nv][0] - pts[k][0]
            dy = pts[(k + 1) % Nv][1] - pts[k][1]
            new.append([pts[k][0] + dx * dt, pts[k][1] + dy * dt])
        dist += math.hypot(new[0][0] - pts[0][0], new[0][1] - pts[0][1])
        pts = new
        ang = math.atan2(pts[0][1], pts[0][0])
        dA = ang - ang_prev
        if dA < -math.pi:
            dA += 2 * math.pi
        ang_total += dA
        ang_prev = ang
    rnum = math.hypot(*pts[0])
    ok1 = ok1 and abs(rnum - math.exp(-av * ang_total)) < 1e-3
    speed = sp.simplify(sp.sqrt(sp.diff(f1, t)**2 + sp.diff(g1, t)**2))
    su = sp.integrate(speed, (t, 0, sp.Symbol("u", positive=True)))
    lim = sp.limit(su, sp.Symbol("u", positive=True), sp.oo)
    ok2 = sp.simplify(lim - sp.sqrt(al**2 + 1) / al) == 0
    ok2 = ok2 and abs(dist + math.sqrt(av**2 + 1) / av * math.exp(-av * ang_total) - math.sqrt(av**2 + 1) / av) < 2e-3
    report(4, ok1 and ok2, f"(1) 접선∥P1P2 확인, n=4 시뮬 r={rnum:.5f} vs e^(-αt)={math.exp(-av*ang_total):.5f}  (2) lim s={lim}")


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
