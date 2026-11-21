#!/usr/bin/env python3
"""KHU-G24D (경희대 2024학년도) — sympy 검산
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "KHU-G24D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import math
    d = sp.symbols("d", positive=True)
    # O1=(-d,0), O2=(0,0), 두 반지름 1. P=두 원의 위쪽 교점, Q=(-1,0), R=(0,sqrt(1-d^2))
    def minval(dv, N=600):
        Px, Py = -dv / 2, math.sqrt(1 - dv * dv / 4)
        aP = math.atan2(Py, Px + dv)          # O1 기준 P 의 각
        bP = math.atan2(Py, Px)               # O2 기준 P 의 각
        best = 9
        for i in range(N + 1):
            al = aP + (math.pi - aP) * i / N  # 호 AP
            for j in range(0, N + 1, 3):
                be = bP + (math.pi - bP) * j / N  # 호 QP
                v = math.hypot(math.cos(al) + math.cos(be), math.sin(al) + math.sin(be))
                best = min(best, v)
        return best
    # 최소는 M=P, N=Q: |u+v|^2 = 2+2cos(angle) , cos = -cos(aP) = -(d/2)
    expr = sp.sqrt(2 - 2 * (d / 2))
    dsol = sp.solve(sp.Eq(expr, 2 * sp.sqrt(3) / 3), d)
    dv = dsol[0]
    ok1 = dv == sp.Rational(2, 3) and abs(minval(float(dv)) - 2 * math.sqrt(3) / 3) < 1e-4
    M = sp.Matrix([-dv / 2, sp.sqrt(1 - dv**2 / 4)])
    R = sp.Matrix([0, sp.sqrt(1 - dv**2)])
    MR2 = sp.nsimplify(sp.expand((M - R).dot(M - R)))
    ok2 = sp.simplify(MR2 - (sp.Rational(14, 9) - 4 * sp.sqrt(10) / 9)) == 0
    report(1, ok1 and ok2, f"(1) O1O2={dv} (격자 최솟값 {minval(float(dv)):.5f})  (2) MR^2={MR2}")


def p02():
    X = sp.symbols("X", real=True)
    b, c, a = sp.symbols("b c a", real=True)
    g = b / 2 * X**2 + X + 1                 # f(0)=f'(0)=1
    h = c / 2 * (X - 2)**2 + (X - 2) + 1     # f(2)=f'(2)=1
    sol = sp.solve([g.subs(X, 1) - h.subs(X, 1), sp.diff(g, X).subs(X, 1) - sp.diff(h, X).subs(X, 1)], [b, c], dict=True)[0]
    g, h = g.subs(sol), h.subs(sol)
    fp1 = sp.diff(g, X).subs(X, 1)
    ok1 = fp1 == -1
    # (2) 기울기 같은 두 점: a<1<a+1/2 (서로 다른 이차식 구간)
    gp, hp = sp.diff(g, X), sp.diff(h, X)
    cand = []
    for f1, f2, cond in [(gp, gp, lambda v: v + sp.Rational(1, 2) <= 1), (gp, hp, lambda v: v < 1 < v + sp.Rational(1, 2)), (hp, hp, lambda v: v >= 1)]:
        for v in sp.solve(sp.Eq(f1.subs(X, a), f2.subs(X, a + sp.Rational(1, 2))), a):
            if cond(v):
                cand.append(v)
    ok2 = cand == [sp.Rational(3, 4)]
    av = cand[0]; x1, x2 = av, av + sp.Rational(1, 2)
    l = g.subs(X, x1) + gp.subs(X, x1) * (X - x1)
    m = h.subs(X, x2) + hp.subs(X, x2) * (X - x2)
    distinct = sp.simplify(l - m) != 0
    S = sp.integrate(l - g, (X, x1, 1)) + sp.integrate(l - h, (X, 1, x2))
    # 다른 해석: x=x2 오른쪽에서 l 과 곡선 사이 영역
    xr = [r for r in sp.solve(sp.Eq(l, h), X) if r > x2][0]
    S_right = sp.simplify(sp.integrate(l - h, (X, x2, xr)))
    ok3 = sp.simplify(48 * S - sp.Rational(3, 2)) == 0 and distinct
    report(2, ok1 and ok2 and ok3, f"(1) g={sp.expand(g)}, h={sp.expand(h)}, f'(1)={fp1}  (2) a={cand}, 48S(왼쪽 영역)={48*S}, [참고] 오른쪽 영역 48S={sp.nsimplify(48*S_right)}")


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
