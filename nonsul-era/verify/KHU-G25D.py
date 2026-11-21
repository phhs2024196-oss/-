#!/usr/bin/env python3
"""KHU-G25D (경희대 2025학년도 논술 의·약학계 수학 실전, KHU-G25D-01~02) — sympy 검산.
규칙: 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import itertools
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "KHU-G25D"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ellipse_axes(f1, f2):
    """x축 위 두 초점, 원점을 지나는 타원의 (반장축, 반단축^2) — 정의(거리의 합)로 직접 계산."""
    a = (sp.Abs(f1) + sp.Abs(f2)) / 2               # 원점에서 두 초점까지 거리의 합 = 2a
    c = sp.Abs(f2 - f1) / 2
    return sp.simplify(a), sp.simplify(a**2 - c**2)


def p01():
    p, tt, k = sp.symbols("p t k", positive=True)
    n = sp.symbols("n", positive=True)
    # (1)
    a1, b1sq = ellipse_axes(p, 3 * p)
    a2, b2sq = ellipse_axes(p, tt)
    sols = sp.solve(sp.Eq(b1sq / a1**2, b2sq / a2**2), tt)   # 단축/장축 비 같음 ⇔ 마름모 닮음
    sols = [s for s in sols if sp.simplify(s - 3 * p) != 0]
    ok1 = len(sols) == 1 and sp.simplify(sols[0] - p / 3) == 0
    # (2) 초점 (p,0),(n,0), 원점 통과: 중심 (A,0), A=(n+p)/2
    A, Bsq = ellipse_axes(p, n)
    X, Y = sp.symbols("X Y", real=True)
    F = (X - A)**2 / A**2 + Y**2 / Bsq - 1
    ypos = sp.sqrt(Bsq * (1 - (k - A)**2 / A**2))
    slope = sp.simplify((-sp.diff(F, X) / sp.diff(F, Y)).subs({X: k, Y: ypos}))
    lim = sp.limit(slope, n, sp.oo)
    ok2 = sp.simplify(lim - sp.sqrt(p / k)) == 0
    # 수치 확인 p=3, k=1, n 큰 값
    val = float(slope.subs({p: 3, k: 1, n: 10**8}))
    ok2 &= abs(val - 3**0.5) < 1e-3
    report(1, ok1 and ok2, f"(1) t={sols}, (2) a_n 극한={sp.simplify(lim)} (p=3,k=1,n=1e8 → {val:.6f})")


def area2(O, A, B):
    return abs((A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]))   # 넓이의 2배


def walk(trials):
    pts = [(0, 0)]
    for coin, die in trials:
        x0, y0 = pts[-1]
        pts.append((x0 + die, y0) if coin == "H" else (x0, y0 + die))
    return pts


def p02():
    outcomes = [(c, d) for c in "HT" for d in range(1, 7)]       # 각 1/12
    # (1)
    good = sum(1 for tr in itertools.product(outcomes, repeat=2) if area2(*[walk(tr)[i] for i in (0, 1, 2)]) >= 12)
    P1 = Fraction(good, 12**2)
    # (2)
    nA = nAB = 0
    for tr in itertools.product(outcomes, repeat=3):
        pts = walk(tr)
        if area2(pts[0], pts[2], pts[3]) >= 36:
            nA += 1
            nAB += tr[2][1] == 6
    P2 = Fraction(nAB, nA)
    PA = Fraction(nA, 12**3)
    ok = P1 == Fraction(17, 72) and P2 == Fraction(19, 32) and PA == Fraction(2, 27)
    report(2, ok, f"(1) P={P1}, (2) P(A)={PA}, P(A∩B)={Fraction(nAB, 12**3)}, P(B|A)={P2} (전수 12²·12³)")


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
