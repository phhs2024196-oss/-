#!/usr/bin/env python3
"""KHU-G22D (경희대 2022학년도 논술 의학계(수학) 11.20 오후, KHU-G22D-01~02) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import math
import re
import sys

import sympy as sp

PREFIX = "KHU-G22D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def transmit(N, r):
    """유리창 N장 사이 다중 반사의 정상 상태: 창 k 와 k+1 사이 오른쪽 흐름 R_k, 왼쪽 흐름 L_k.
    창 k: 왼쪽 입사 R_{k-1}(창 1은 a=1), 오른쪽 입사 L_k → 오른쪽 출력 R_k = r R_{k-1} + (1-r) L_k,
    왼쪽 출력 L_{k-1} = (1-r) R_{k-1} + r L_k. 마지막 창 오른쪽에서 들어오는 빛은 0."""
    Rs = sp.symbols(f"R0:{N + 1}")
    Ls = sp.symbols(f"L0:{N + 1}")
    eqs = [sp.Eq(Rs[0], 1), sp.Eq(Ls[N], 0)]
    for k in range(1, N + 1):
        eqs.append(sp.Eq(Rs[k], r * Rs[k - 1] + (1 - r) * Ls[k]))
        eqs.append(sp.Eq(Ls[k - 1], (1 - r) * Rs[k - 1] + r * Ls[k]))
    sol = sp.solve(eqs, list(Rs) + list(Ls), dict=True)[0]
    return sp.simplify(sol[Rs[N]])


def p01():
    """논제 I-1 유리창 다중 반사"""
    r = sp.symbols("r", positive=True)
    T2 = transmit(2, r)
    roots = [v for v in sp.solve(sp.Eq(T2, sp.Rational(1, 2)), r) if 0 < v < 1]
    ok1 = roots == [sp.Rational(2, 3)]
    rv = roots[0]
    ok2 = all(sp.simplify(transmit(N, rv) - sp.Rational(2, N + 2)) == 0 for N in range(1, 9))   # md: p_n = 2a/(n+2)
    # 등비급수 직접 합(공식 해설 방식)도 일치
    ok1 = ok1 and sp.simplify(T2 - r**2 / (1 - (1 - r)**2)) == 0
    report(1, ok1 and ok2, f"(1) T2={sp.factor(T2)}, r={roots}  (2) N=1..8 투과율 = 2/(N+2) 확인")


def p02():
    """논제 I-2 (1) y축까지 거리 (2) AB=1 극한"""
    al = sp.symbols("alpha", positive=True)
    f1, g1 = sp.exp(-al * t) * sp.cos(t), sp.exp(-al * t) * sp.sin(t)
    speed = sp.simplify(sp.sqrt(sp.diff(f1, t)**2 + sp.diff(g1, t)**2))
    D = sp.simplify(sp.integrate(speed, (t, 0, sp.pi / 2)))
    ok1 = sp.simplify(D - sp.sqrt(al**2 + 1) / al * (1 - sp.exp(-sp.pi * al / 2))) == 0
    # 수치 추적 시뮬레이션(n=3, 5): x1 이 처음 0 이 될 때까지의 거리
    for Nv in (3, 5):
        av = (1 - math.cos(2 * math.pi / Nv)) / math.sin(2 * math.pi / Nv)
        pts = [[math.cos(2 * math.pi * k / Nv), math.sin(2 * math.pi * k / Nv)] for k in range(Nv)]
        dt, dist = 2e-5, 0.0
        while pts[0][0] > 0:
            new = [[pts[k][0] + (pts[(k + 1) % Nv][0] - pts[k][0]) * dt,
                    pts[k][1] + (pts[(k + 1) % Nv][1] - pts[k][1]) * dt] for k in range(Nv)]
            dist += math.hypot(new[0][0] - pts[0][0], new[0][1] - pts[0][1])
            pts = new
        ok1 = ok1 and abs(dist - float(D.subs(al, av))) < 1e-3
    # (2) AB=1, S=(a+b)^2/(2ab): b 를 이분법으로 구해 a→∞, a→0 수치 경향
    import mpmath as mp
    mp.mp.dps = 60

    def S_of(av, branch):
        av = mp.mpf(av)
        g = lambda bv: mp.sqrt((av - bv)**2 + (1 / av - 1 / bv)**2) - 1
        lo, hi = (av, av + 2) if branch > 0 else (av / 2, av)
        for _ in range(400):
            mid = (lo + hi) / 2
            if (g(mid) < 0) == (branch > 0):
                lo = mid
            else:
                hi = mid
        bv = (lo + hi) / 2
        return float((av + bv)**2 / (2 * av * bv))
    vals = [S_of(1e5, 1), S_of(1e5, -1), S_of(1e-4, 1), S_of(1e-4, -1)]
    ok2 = all(abs(v - 2) < 1e-4 for v in vals)
    report(2, ok1 and ok2, f"(1) 거리={D}  (2) S≈{[round(v, 7) for v in vals]} → 2")


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
