#!/usr/bin/env python3
"""KHU-G21A (경희대 2021학년도 논술 자연계 12.05 오전, 수학 논제 I-1·I-2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "KHU-G21A"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[논제 I-1] AB=100, EF=120, S1은 AB·BC, S2는 CD·AD에 접함, 두 원은 T에서 외접."""
    r1, r2 = sp.symbols("r1 r2", positive=True)
    # 좌표: D(0,0), A(0,H), B(100,H), C(100,0). O1=(100-r1, H-r1), O2=(r2, r2)
    H = sp.symbols("H", positive=True)
    O1 = sp.Matrix([100 - r1, H - r1]); O2 = sp.Matrix([r2, r2])
    T = O2 + (O1 - O2) * r2 / (r1 + r2)
    # T를 지나는 수직선(AD에 평행) 위의 E(원 S1), F(원 S2): E=(Tx, 2*O1y - Ty), F=(Tx, 2*O2y - Ty)
    E_y = 2 * O1[1] - T[1]; F_y = 2 * O2[1] - T[1]
    eqs = [sp.Eq((O1 - O2).dot(O1 - O2), (r1 + r2) ** 2), sp.Eq(sp.simplify(E_y - F_y), 120)]
    # 두 식에서 r1+r2 와 H 결정: 수평거리 k=100-r1-r2, 수직거리 = (E_y-F_y)/2
    s, k = sp.symbols("s k", positive=True)
    sol = sp.solve([sp.Eq(s**2, k**2 + 60**2), sp.Eq(s + k, 100)], [s, k], dict=True)
    ok1 = False; area = None
    for so in sol:
        S = so[s]
        Hval = S + 60   # BC = r1+r2 + (수직거리 60)
        # 실제 좌표로 확인: r1=S/2, r2=S/2 에서 두 식 성립
        sub = {r1: S / 2, r2: S / 2, H: Hval}
        c1 = sp.simplify(eqs[0].lhs.subs(sub) - eqs[0].rhs.subs(sub)) == 0
        c2 = sp.simplify(E_y.subs(sub) - F_y.subs(sub) - 120) == 0
        if c1 and c2:
            area = 100 * Hval; ok1 = area == 12800
    # (2) r1+r2=68, 0<=r_i<=50 (가로 100이 지름 제한) → 18<=r1<=50
    x = sp.symbols("x", real=True)
    Ssum = sp.pi * (x**2 + (68 - x) ** 2)
    lo, hi = 68 - 50, 50
    crit = [c for c in sp.solve(sp.diff(Ssum, x), x) if lo <= c <= hi]
    vals = {c: Ssum.subs(x, c) for c in crit + [lo, hi]}
    mn = min(vals.values()); mx = max(vals.values())
    ok2 = mn == 2312 * sp.pi and mx == 2824 * sp.pi and crit == [34]
    # 대표 오답: 반지름 범위를 0<=r1<=68 로 잡으면 최댓값 68^2 π
    wrong = Ssum.subs(x, 0)
    report(1, ok1 and ok2, f"넓이={area}, 최소={mn}(r=34), 최대={mx}(r=18,50); 범위 오류 시 {wrong}")


def p02():
    """[논제 I-2] 정사각형 A(-1,2),B(1,2),C(1,0),D(-1,0), 반원 중심 O, T(-sinθ, cosθ)."""
    th = sp.symbols("theta", real=True)
    # 접선: -sinθ x + cosθ y = 1  → y = tanθ x + secθ
    line_y = lambda X: (1 + sp.sin(th) * X) / sp.cos(th)
    # (1) B(1,2)를 지남: 2cosθ = 1 + sinθ, 0<=θ<π/2
    sols = [s for s in sp.solve(sp.Eq(2 * sp.cos(th), 1 + sp.sin(th)), th) if 0 <= float(s) < float(sp.pi / 2)]
    al = sols[0]
    ok1 = (sp.nsimplify(sp.sin(al)), sp.nsimplify(sp.cos(al)), sp.nsimplify(sp.tan(al))) == (sp.Rational(3, 5), sp.Rational(4, 5), sp.Rational(3, 4))
    # (2) 직선과 정사각형 경계의 두 교점 사이 거리를 직접 계산(수치)해 f(θ)와 비교
    def chord(t):
        t = float(t)
        import math
        pts = []
        m, b = math.tan(t), 1 / math.cos(t)
        for X in (-1.0, 1.0):
            Y = m * X + b
            if -1e-12 <= Y <= 2 + 1e-12: pts.append((X, Y))
        if m > 1e-15:
            for Y in (0.0, 2.0):
                X = (Y - b) / m
                if -1 - 1e-12 <= X <= 1 + 1e-12: pts.append((X, Y))
        uniq = []
        for p in pts:
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1e-9 for q in uniq): uniq.append(p)
        (x1, y1), (x2, y2) = uniq[:2]
        return math.hypot(x1 - x2, y1 - y2)
    alf = float(al)
    f1 = 2 / sp.cos(th)
    f2 = 2 / sp.sin(th) + 1 / sp.cos(th) - 1 / (sp.sin(th) * sp.cos(th))
    ok2 = True
    import math
    for i in range(1, 400):
        t = (math.pi / 2) * i / 400
        f = f1 if t <= alf else f2
        if abs(float(f.subs(th, t)) - chord(t)) > 1e-8: ok2 = False
    # (3) J = ∫_0^{π/4} (f sinθ cosθ - cosθ) dθ, sin J
    J = sp.integrate(sp.simplify(f1 * sp.sin(th) * sp.cos(th) - sp.cos(th)), (th, 0, al)) + \
        sp.integrate(sp.simplify(f2 * sp.sin(th) * sp.cos(th) - sp.cos(th)), (th, al, sp.pi / 4))
    sJ = sp.nsimplify(sp.simplify(sp.expand_trig(sp.sin(J))))
    ok3 = sp.simplify(sJ + sp.sqrt(2) / 10) == 0
    report(2, ok1 and ok2 and ok3, f"α: sin,cos,tan=3/5,4/5,3/4; f(θ) 격자 399점 일치={ok2}; sin J={sJ}")


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
