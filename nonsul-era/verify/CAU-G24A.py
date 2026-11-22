#!/usr/bin/env python3
"""CAU-G24A (2024학년도 중앙대 수시 논술 자연계열Ⅰ) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """주사위·주머니 전수 → 원뿔(r=5,h=10) 내접 원기둥(반지름 k-3) 부피 기댓값 (답 24π)."""
    import itertools
    rr = sp.symbols("rr", positive=True)
    # 내접 원기둥 높이: 원뿔 모선 위 점 (rr, h) → h = 10(1 - rr/5) (닮음)
    hgt = 10 * (1 - rr / 5)
    E = sp.Integer(0)
    for die in range(1, 7):
        if die % 3 == 0:
            draws = [(sum(a), b) for a in itertools.combinations([1, 2, 3], 2) for b in [1, 2]]
        else:
            draws = [(a, sum(b)) for a in [1, 2, 3] for b in itertools.combinations([1, 2], 2)]
        for sa, sb in draws:
            k = sa + sb
            vol = sp.pi * (k - 3)**2 * hgt.subs(rr, k - 3)
            E += sp.Rational(1, 6) * sp.Rational(1, len(draws)) * vol
    report(1, sp.simplify(E - 24 * sp.pi) == 0, f"E[부피]={sp.simplify(E)}")


def p02():
    """(1) Σ_{k=2}^{9}(k^{k ln2/ln k} - 2^{2 ln k/ln2}) 고정밀 직접 계산 (답 736).
    (2) 각 조건으로 B 를 수치로 찾아 리만합 → (6√3/π)ln(3/2)."""
    import mpmath as mp
    mp.mp.dps = 40
    S = sum(mp.power(k, k * mp.log(2) / mp.log(k)) - mp.power(2, 2 * mp.log(k) / mp.log(2)) for k in range(2, 10))
    ok1 = abs(S - 736) < 1e-25

    def f_geo(tt):
        # A=(1,tan t). B=(b,0), b≥1, ∠OAB = 2π/3 이 되는 b 를 이분법으로
        Ax, Ay = mp.mpf(1), mp.tan(tt)
        def ang(b):
            u = (-Ax, -Ay); v = (b - Ax, -Ay)
            return mp.acos((u[0] * v[0] + u[1] * v[1]) / (mp.sqrt(u[0]**2 + u[1]**2) * mp.sqrt(v[0]**2 + v[1]**2)))
        lo, hi = mp.mpf(1), mp.mpf(50)          # ang 은 b 에 대해 증가
        for _ in range(90):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ang(mid) < 2 * mp.pi / 3 else (lo, mid)
        return (lo + hi) / 2
    mp.mp.dps = 20
    integ = mp.quad(f_geo, [0, mp.pi / 6])
    lim = 6 / mp.pi * integ                       # (1/n)Σ f(πk/6n) → (6/π)∫_0^{π/6} f
    target = 6 * mp.sqrt(3) / mp.pi * mp.log(mp.mpf(3) / 2)
    Nn = 400
    riem = sum(f_geo(mp.pi * k / (6 * Nn)) for k in range(1, Nn + 1)) / Nn
    ok2 = abs(lim - target) < 1e-12 and abs(riem - target) < 5e-3
    report(2, ok1 and ok2, f"(1) 합={mp.nstr(S, 20)}  (2) 극한={mp.nstr(lim, 15)} (n=400 리만합 {mp.nstr(riem, 8)})")


def p03():
    """(1) |y| 의 수치적분 (답 13/2). (2) P(x,√(x+2)) 격자·임계점으로 tan²∠APB 최댓값 (답 (11+5√5)/2)."""
    import mpmath as mp
    mp.mp.dps = 30
    yf = lambda X: X / (X**2 + 1) * (mp.log(X**2 + 1)**2 - 6 * mp.log(X**2 + 1) + 5)
    r1, r2 = mp.sqrt(mp.e - 1), mp.sqrt(mp.e**5 - 1)
    area = mp.quad(lambda X: abs(yf(X)), [0, r1, r2])
    ok1 = abs(area - mp.mpf(13) / 2) < 1e-20 and abs(yf(r1)) < 1e-20 and abs(yf(r2)) < 1e-20
    # (2)
    px, py = x, sp.sqrt(x + 2)
    u = sp.Matrix([-1 - px, -py]); v = sp.Matrix([1 - px, -py])
    cosv = u.dot(v); sinv = u[0] * v[1] - u[1] * v[0]
    tan2 = sp.simplify(sinv**2 / cosv**2)
    import numpy as np
    g = sp.lambdify(x, tan2, "numpy")
    xs = np.linspace(-1, 1, 400001)
    # θ 가 둔각인지 확인(cos 부호): cos<0 이면 tanθ<0 이지만 tan² 은 그대로
    cosn = sp.lambdify(x, cosv, "numpy")(xs)
    gm = float(np.max(g(xs)))
    crit = [c for c in sp.solveset(sp.diff(tan2, x), x, sp.Interval(-1, 1))]
    ex = max([tan2.subs(x, c) for c in crit] + [tan2.subs(x, -1), tan2.subs(x, 1)], key=lambda e: float(e))
    ok2 = sp.simplify(ex - (11 + 5 * sp.sqrt(5)) / 2) == 0 and abs(gm - float(ex)) < 1e-6 and bool(np.all(cosn > 0))
    report(3, ok1 and ok2, f"(1) 넓이={mp.nstr(area, 15)}  (2) max tan²θ={sp.nsimplify(ex)} (격자 {gm:.8f}, θ 항상 예각)")


def p04():
    """(1) 정삼각기둥 좌표로 삼각형 PFQ 외접원 반지름 (답 √39/6).
    (2) 포물선·직선 교점으로 둘레 40 → m, 사각형 넓이 (답 64)."""
    s3 = sp.sqrt(3)
    # 윗면 ABC(z=2), 아랫면 DEF(z=0); A 위 D, B 위 E, C 위 F. 한 변 2
    B = sp.Matrix([0, 0, 2]); C = sp.Matrix([2, 0, 2]); A = sp.Matrix([1, s3, 2])
    E = sp.Matrix([0, 0, 0]); F = sp.Matrix([2, 0, 0]); D = sp.Matrix([1, s3, 0])
    M = (A + C) / 2; N = (B + E) / 2
    P = (M + N) / 2; Q = (C + N) / 2
    a = (Q - F).norm(); b = (P - F).norm(); c = (P - Q).norm()
    area = ((P - F).cross(Q - F)).norm() / 2
    Rr = sp.simplify(a * b * c / (4 * area))
    ok1 = sp.simplify(Rr - sp.sqrt(39) / 6) == 0
    # (2)
    m = sp.symbols("m", positive=True)
    X = sp.symbols("X")
    roots = sp.solve(sp.Eq((m * (X - 1))**2, 4 * X), X)
    pts = [(r, m * (r - 1)) for r in roots]
    (x1, y1), (x2, y2) = pts
    PQ = sp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    per = PQ + (x1 + 1) + (x2 + 1) + sp.Abs(y1 - y2)
    msol = sp.nsolve(per - 40, m, 0.5)
    vals = {m: msol}
    poly = [(float((x1).subs(vals)), float(y1.subs(vals))), (-1.0, float(y1.subs(vals))),
            (-1.0, float(y2.subs(vals))), (float(x2.subs(vals)), float(y2.subs(vals)))]
    sh = 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % 4][1] - poly[(i + 1) % 4][0] * poly[i][1] for i in range(4)))
    # 둘레 함수가 m>0 에서 단조인지(해 유일) 확인
    import numpy as np
    pf = sp.lambdify(m, per, "numpy")
    ms = np.linspace(0.05, 20, 4000)
    mono = bool(np.all(np.diff(pf(ms)) < 0))
    ok2 = abs(sh - 64) < 1e-9 and mono and abs(float(msol)**2 - 1 / 3) < 1e-12
    report(4, ok1 and ok2, f"(1) R={Rr}  (2) m²={float(msol)**2:.12f}, 넓이={sh:.10f}, 둘레 단조감소={mono}")


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
