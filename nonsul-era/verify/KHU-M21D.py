#!/usr/bin/env python3
"""KHU-M21D (경희대 2021학년도 모의논술 의학계 수학, KHU-M21D-01~05) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import random
import re
import sys

import sympy as sp

PREFIX = "KHU-M21D"
WIDTH = 2
RESULTS = []
x, y, t, n = sp.symbols("x y t n", real=True)
T = sp.Symbol("T", nonnegative=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


_QSYM = None


def Q_of(tv):
    """P(-1,0), (0,t) 직선과 단위원의 다른 교점 — 설정에서 직접 계산(기호 해를 한 번 구해 대입)."""
    global _QSYM
    if _QSYM is None:
        sols = sp.solve([x**2 + y**2 - 1, y - T * (x + 1)], [x, y], dict=True)
        other = [s for s in sols if sp.simplify(s[x] + 1) != 0]
        _QSYM = (sp.simplify(other[0][x]), sp.simplify(other[0][y]))
    return sp.simplify(_QSYM[0].subs(T, tv)), sp.simplify(_QSYM[1].subs(T, tv))


def p01():
    qx, qy = Q_of(T)
    ok = sp.simplify(qx - (1 - T**2) / (1 + T**2)) == 0 and sp.simplify(qy - 2 * T / (1 + T**2)) == 0
    report(1, ok, f"Q = ({qx}, {qy})")


def p02():
    qx, qy = Q_of(T)
    slope = sp.simplify(sp.diff(qy, T) / sp.diff(qx, T))
    ok = sp.simplify(slope - (T**2 - 1) / (2 * T)) == 0
    # 접선 = 원의 접선 x*qx + y*qy = 1 과 일치하는지 (t≠0), md 식 비교
    md_line = (T**2 - 1) / (2 * T) * (x - (1 - T**2) / (1 + T**2)) + 2 * T / (1 + T**2)
    geo = sp.solve(x * qx + y * qy - 1, y)[0]
    ok = ok and sp.simplify(md_line - geo) == 0
    # t=0: dx/dt=0, Q=(1,0), 접선 x=1 ; 동치식 (1-t^2)x+2ty=1+t^2
    ok = ok and sp.diff(qx, T).subs(T, 0) == 0 and sp.simplify(((1 - T**2) * x + 2 * T * y - (1 + T**2)).subs({T: 0, x: 1})) == 0
    report(2, ok, f"dy/dx={slope}, 접선 식이 원의 접선 xx_Q+yy_Q=1과 일치, t=0이면 x=1")


def p03():
    t1, t2 = sp.symbols("t1 t2", nonnegative=True)
    ok = True
    for _ in range(20):
        a, b = sorted(random.uniform(0, 1) for _ in range(2))
        # 각을 Q 좌표로부터 직접 계산
        def ang(tv):
            qx, qy = Q_of(tv)
            return sp.atan2(qy, qx + 1)  # ∠OPQ: P에서 Q를 보는 각
        val = sp.N(sp.sin(ang(b) - ang(a)), 30)
        md = sp.N((b - a) / sp.sqrt((b**2 + 1) * (a**2 + 1)), 30)
        if abs(val - md) > 1e-9:
            ok = False
    # 기호 확인
    expr = sp.sin(sp.atan(t2) - sp.atan(t1))
    ok = ok and sp.simplify(sp.expand_trig(expr) - (t2 - t1) / sp.sqrt((t2**2 + 1) * (t1**2 + 1))) == 0
    report(3, ok, "sin(θ2-θ1)=(t2-t1)/√((t1²+1)(t2²+1)) — 무작위 20점 + 기호 확인")


def area_tri(P, A, B):
    return sp.Abs((A[0] - P[0]) * (B[1] - P[1]) - (B[0] - P[0]) * (A[1] - P[1])) / 2


def p04():
    k = sp.Symbol("k", integer=True, nonnegative=True)
    nn = sp.Symbol("nn", positive=True)
    P = (-1, 0)
    ok = True
    for N in [1, 2, 3, 5, 8]:
        for kk in range(N):
            A = Q_of(sp.Rational(kk, N)); B = Q_of(sp.Rational(kk + 1, N))
            direct = sp.nsimplify(area_tri(P, A, B))
            md = sp.Rational(2, N) / (((sp.Rational(kk + 1, N)) ** 2 + 1) * ((sp.Rational(kk, N)) ** 2 + 1))
            if sp.simplify(direct - md) != 0:
                ok = False
    # S_n 극한: 다각형 넓이 → 직각이등변삼각형 + 사분원 (기하적으로 독립 계산)
    target = sp.Rational(1, 2) + sp.integrate(sp.sqrt(1 - x**2), (x, 0, 1))
    S = lambda N: sum((2 / N) / ((((kk + 1) / N) ** 2 + 1) * ((kk / N) ** 2 + 1)) for kk in range(N))
    s2000 = S(2000)
    ok = ok and abs(s2000 - sp.N(target)) < 1e-3 and sp.simplify(target - (sp.Rational(1, 2) + sp.pi / 4)) == 0
    report(4, ok, f"a_k 좌표로 직접 계산과 일치(n=1..8), S_2000={float(s2000):.6f} → 1/2+π/4={float(sp.N(target)):.6f}")


def p05():
    I = sp.integrate(2 / (x**2 + 1) ** 2, (x, 0, 1))
    ok = sp.simplify(I - (sp.Rational(1, 2) + sp.pi / 4)) == 0
    for N in [10, 100, 1000]:
        Tn = sum(sp.Rational(2, N) / ((sp.Rational(kk, N) ** 2 + 1) ** 2) for kk in range(N))
        Sn = sum(sp.Rational(2, N) / ((sp.Rational(kk + 1, N) ** 2 + 1) * (sp.Rational(kk, N) ** 2 + 1)) for kk in range(N))
        d = Tn - Sn
        bound = sp.Rational(2, N**3) * sum(2 * kk + 1 for kk in range(N))
        if not (0 <= d <= bound and bound == sp.Rational(2, N)):
            ok = False
    report(5, ok, f"∫=1/2+π/4 (sympy: {sp.simplify(I)}), 0≤T_n−S_n≤2/n (n=10,100,1000) ⇒ 극한 0 [증명-수치확인]")


if __name__ == "__main__":
    random.seed(2021)
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
