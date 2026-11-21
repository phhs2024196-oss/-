#!/usr/bin/env python3
"""DGU-G24 (2024학년도 동국대 논술 실전 기출) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G24"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """x²+y²=(kx+1)²: (1) 원 (2) 타원 (3) 포물선 (4) 쌍곡선 — 초점·꼭짓점·준선·점근선을 정의로 검산."""
    import random
    R = sp.Rational
    X, Y = sp.symbols("X Y", real=True)
    def curve(k):
        return sp.expand(X**2 + Y**2 - (k * X + 1)**2)
    def pts(k, xs):
        out = []
        for xv in xs:
            for yv in sp.solve(curve(k).subs(X, xv), Y):
                if yv.is_real:
                    out.append((xv, yv))
        return out
    dist = lambda p, q: sp.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
    # (1) k=0: 중심 (0,0), 반지름 1
    ok1 = all(sp.simplify(dist(p, (0, 0)) - 1) == 0 for p in pts(0, [R(-1, 2), 0, R(1, 3)]))
    # (2) k=1/2: 초점 (0,0),(4/3,0), 꼭짓점 (2,0),(-2/3,0),(2/3,±2√3/3) — 거리의 합 = 장축 길이 8/3
    F1, F2 = (0, 0), (R(4, 3), 0)
    e_pts = pts(R(1, 2), [R(random.randint(-60, 190), 100) for _ in range(8)])
    V = [(2, 0), (R(-2, 3), 0), (R(2, 3), 2 * sp.sqrt(3) / 3), (R(2, 3), -2 * sp.sqrt(3) / 3)]
    ok2 = all(sp.simplify(dist(p, F1) + dist(p, F2) - R(8, 3)) == 0 for p in e_pts)         and all(sp.simplify(curve(R(1, 2)).subs({X: v[0], Y: v[1]})) == 0 for v in V)         and sp.solve(sp.diff(sp.solve(curve(R(1, 2)), Y)[1]**2, X), X) == [R(2, 3)]
    # (3) k=1: 초점 (0,0), 준선 x=-1 — 초점까지 거리 = 준선까지 거리
    p_pts = pts(1, [R(-1, 2), 0, 1, R(7, 3), 5])
    ok3 = all(sp.simplify(dist(p, (0, 0)) - (p[0] + 1)) == 0 for p in p_pts)
    # (4) k=2: 초점 (0,0),(-4/3,0), 거리의 차 = 2/3, 점근선 y=±√3(x+2/3)
    h_pts = pts(2, [R(1, 2), 1, 3, R(-2, 1), R(-5, 2)])
    ok4a = all(sp.simplify(sp.Abs(dist(p, (0, 0)) - dist(p, (R(-4, 3), 0))) - R(2, 3)) == 0 for p in h_pts)
    yb = sp.solve(curve(2), Y)
    yb = [b for b in yb if sp.limit(b / X, X, sp.oo).is_positive][0]
    slope = sp.limit(yb / X, X, sp.oo)
    icpt = sp.limit(yb - slope * X, X, sp.oo)
    ok4 = ok4a and slope == sp.sqrt(3) and sp.simplify(icpt - 2 * sp.sqrt(3) / 3) == 0
    report(1, ok1 and ok2 and ok3 and ok4, f"(1)원 OK (2)합=8/3 확인 {len(e_pts)}점 (3)초점·준선 {len(p_pts)}점 (4)차=2/3, 점근선 y={slope}x+{icpt}")


def p02():
    """(1) g(π/2)=15/8 (2) 최솟값 4/3 (3) 31π/2 — 기울기의 각으로 독립 계산."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    T = sp.symbols("T", real=True)
    f = sp.sin(T) + 4 * T + 1
    m1 = sp.diff(f, T)
    m2 = 1 / m1                                      # 역함수 접선 기울기
    # 두 직선이 이루는 예각의 tan = |(m1-m2)/(1+m1 m2)|
    g = sp.simplify(sp.Abs((m1 - m2) / (1 + m1 * m2)))
    ok1 = sp.simplify(g.subs(T, sp.pi / 2) - sp.Rational(15, 8)) == 0
    gn = sp.lambdify(T, g, "numpy")
    grid = np.linspace(0, 2 * np.pi, 400001)
    mn = gn(grid).min()
    ok2 = abs(mn - 4 / 3) < 1e-9 and sp.simplify(g.subs(T, sp.pi) - sp.Rational(4, 3)) == 0
    fn = sp.lambdify(T, f)
    finv = lambda y: brentq(lambda s: fn(s) - y, -2, 9)
    num = quad(lambda y: gn(finv(y)), fn(0), fn(2 * np.pi), limit=200)[0]
    sym = sp.integrate(sp.simplify(((m1)**2 - 1) / (2 * m1) * m1), (T, 0, 2 * sp.pi))
    ok3 = abs(num - 31 * np.pi / 2) < 1e-7 and sp.simplify(sym - 31 * sp.pi / 2) == 0
    report(2, ok1 and ok2 and ok3, f"g(π/2)={g.subs(T, sp.pi/2)}, 격자 최솟값={mn:.10f}, (3) 기호={sym}, 수치={num:.10f}")


def p03():
    """(1) Q 속력 = 1+2/(a²-1) (2) a=2, 5/3 — 교점 Q의 각 s(t) 를 직접 미분."""
    import numpy as np
    a, t = sp.symbols("a t", positive=True)
    # 직선 AP 위의 점 A + λ(P-A) 가 원 위: λ=1 이 P, 다른 근이 Q (근과 계수: λ1λ2 = (a²-1)/|P-A|²)
    Px, Py = sp.cos(t), sp.sin(t)
    L2 = (Px - a)**2 + Py**2
    lam = (a**2 - 1) / L2
    Qx, Qy = a + lam * (Px - a), lam * Py
    speed = sp.sqrt(sp.diff(Qx, t)**2 + sp.diff(Qy, t)**2)
    claim = 1 + 2 / (a**2 - 1)
    ok1 = True
    t0s = []
    for av in np.linspace(2, 3, 21):
        t0 = sp.nsolve(Qx.subs(a, av), t, 0.8)          # Q=(0,1) 인 시각
        t0s.append(float(t0))
        ok1 &= abs(float(Qy.subs({a: av, t: t0})) - 1) < 1e-12
        ok1 &= abs(float(speed.subs({a: av, t: t0})) - float(claim.subs(a, av))) < 1e-10
    ok_t0 = all(0 <= v < np.pi / 3 for v in t0s)
    ok2 = sp.solve(sp.diff(claim, a), a) == [] and claim.subs(a, 2) == sp.Rational(5, 3) and claim.subs(a, 2) > claim.subs(a, 3)
    report(3, ok1 and ok_t0 and ok2, f"수치 21점 속력 일치, t0∈[{min(t0s):.4f},{max(t0s):.4f}]<π/3, 최대 a=2 → {claim.subs(a, 2)}")


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
