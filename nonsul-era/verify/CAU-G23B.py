#!/usr/bin/env python3
"""CAU-G23B (2023학년도 중앙대 수시 논술 자연계열Ⅱ 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """4×4 격자, 가로·세로(1000원, 60분), 대각선은 (i,i)→(i+1,i+1)만(2400원, 30분). 비용≤9000, 시간≤7시간 경로 전수 → 94."""
    from functools import lru_cache

    @lru_cache(None)
    def paths(i, j):
        if (i, j) == (4, 4):
            return [(0, 0)]
        out = []
        if i < 4:
            out += [(c + 1000, m + 60) for c, m in paths(i + 1, j)]
        if j < 4:
            out += [(c + 1000, m + 60) for c, m in paths(i, j + 1)]
        if i == j and i < 4:
            out += [(c + 2400, m + 30) for c, m in paths(i + 1, j + 1)]
        return out

    allp = paths(0, 0)
    good = [p for p in allp if p[0] <= 9000 and p[1] <= 420]
    # 대각선 수별 분포
    from collections import Counter
    byz = Counter()
    for c, m in good:
        for z in range(5):
            if c == (8 - 2 * z) * 1000 + 2400 * z:
                byz[z] += 1
    report(1, len(good) == 94 and byz == {2: 30, 1: 64}, f"전체 경로 {len(allp)}, 조건 만족 {len(good)}, 대각선 수별 {dict(byz)}")


def p02():
    """(1) ∫_{-π/4}^{π/4} |x|/(1-sinx) dx / (2) P(0,109)에서 y=4(1+x^2/36)^{3/2} 까지 거리 최소 두 점 사이 곡선 길이."""
    I = sp.integrate(-x / (1 - sp.sin(x)), (x, -sp.pi / 4, 0)) + sp.integrate(x / (1 - sp.sin(x)), (x, 0, sp.pi / 4))
    import mpmath as mp
    mp.mp.dps = 30
    In = mp.quad(lambda u: abs(u) / (1 - mp.sin(u)), [-mp.pi / 4, 0, mp.pi / 4])
    ok1 = abs(In - (mp.pi / 2 - mp.log(2))) < 1e-25
    y = 4 * (1 + x ** 2 / 36) ** sp.Rational(3, 2)
    D = x ** 2 + (y - 109) ** 2
    dD = sp.diff(D, x)
    # 수치로 임계점 탐색 + 최소점
    Df = sp.lambdify(x, D, "mpmath")
    dDf = sp.lambdify(x, dD, "mpmath")
    crit = set()
    grid = [mp.mpf(k) / 10 for k in range(-400, 401)]
    for a, b in zip(grid, grid[1:]):
        if dDf(a) == 0:
            crit.add(mp.nstr(a, 20))
        elif dDf(a) * dDf(b) < 0:
            crit.add(mp.nstr(mp.findroot(dDf, (a, b), solver="bisect"), 20))
    crit = sorted(mp.mpf(c) for c in crit)
    vals = [Df(c) for c in crit]
    m = min(vals)
    mins = [c for c, v in zip(crit, vals) if abs(v - m) < 1e-15]
    x1, x2 = mins[0], mins[-1]
    Lnum = mp.quad(lambda u: mp.sqrt(1 + sp.lambdify(x, sp.diff(y, x), "mpmath")(u) ** 2), [x1, 0, x2])
    ok2 = len(mins) == 2 and abs(x2 - 12 * mp.sqrt(2)) < 1e-15 and abs(Lnum - 152 * mp.sqrt(2)) < 1e-15
    # 기호 확인: 1+(y')^2 = (1+x^2/18)^2
    ok3 = sp.simplify(1 + sp.diff(y, x) ** 2 - (1 + x ** 2 / 18) ** 2) == 0
    report(2, ok1 and ok2 and ok3, f"(1) 수치 {mp.nstr(In, 15)} vs π/2-ln2, 기호 {sp.simplify(I)}; (2) 임계점 {[mp.nstr(c, 8) for c in crit]}, 최소 x={mp.nstr(x1, 10)},{mp.nstr(x2, 10)}, 길이 {mp.nstr(Lnum, 15)} (152√2={mp.nstr(152*mp.sqrt(2), 15)})")


def p03():
    """(1) 포물선 y=(a+3/2)^2-(a+1)^2x^2 과 원 x^2+(y-b)^2=b^2 이 서로 다른 두 점에서 만나는 b=f(a), 최댓값 /
    (2) A(cos t, sin t), B(x,√x) 거리 1 유지, B는 t=0 에 1사분면 → t=π/2 의 dx/dt, x 최댓값."""
    a, b, yy = sp.symbols("a b yy", real=True)
    # x^2 소거 → y 에 대한 이차식, 두 점(대칭쌍)에서 만나려면 y 가 중근
    xx2 = ((a + sp.Rational(3, 2)) ** 2 - yy) / (a + 1) ** 2
    q = sp.expand((xx2 + (yy - b) ** 2 - b ** 2) * (a + 1) ** 2)
    disc = sp.discriminant(q, yy)
    sols = sp.solve(sp.Eq(disc, 0), b)
    # 조건 0<2b<(a+3/2)^2, a>=0 에서 유효한 해 선택(수치 확인 a=0.7)
    def valid(expr, av):
        bv = expr.subs(a, av)
        if not bv.is_real:
            return False
        bv = float(bv)
        if not (0 < 2 * bv < (av + 1.5) ** 2):
            return False
        # 중근 y 가 (0, (a+3/2)^2) 안 → x^2>0 인 두 점
        ystar = float(sp.solve(sp.diff(q, yy).subs({a: av, b: bv}), yy)[0])
        return 0 <= ystar < (av + 1.5) ** 2 - 1e-12
    good = [s for s in sols if all(valid(s, av) for av in (0, 0.7, 1, 3))]
    fa = sp.simplify(good[0])
    crit = [r for r in sp.solve(sp.diff(fa, a), a) if r.is_real and r >= 0]
    cand = [fa.subs(a, 0)] + [fa.subs(a, r) for r in crit] + [sp.limit(fa, a, sp.oo)]
    mx = max(cand)
    ok1 = len(good) == 1 and sp.simplify(fa - (2 * a ** 2 + 5 * a + 2) / (2 * a ** 2 + 4 * a + 2)) == 0 and mx == sp.Rational(9, 8)
    # (2) 연속 추적: F(x,t)=(x-cos t)^2+(√x - sin t)^2-1=0
    import mpmath as mp
    mp.mp.dps = 30
    F = lambda X, T: (X - mp.cos(T)) ** 2 + (mp.sqrt(X) - mp.sin(T)) ** 2 - 1
    # t=0 출발점: 1사분면 근
    X = mp.findroot(lambda X: F(X, 0), 0.9)
    N = 4000
    xs = [X]
    for k in range(1, N + 1):
        T = mp.pi * k / N
        X = mp.findroot(lambda Z: F(Z, T), X)
        xs.append(X)
    xmax = max(xs)
    # t=π/2 에서 dx/dt (음함수 미분: -F_t/F_x)
    Xh = xs[N // 2]
    Th = mp.pi / 2
    Fx = mp.diff(lambda Z: F(Z, Th), Xh)
    Ft = mp.diff(lambda T: F(Xh, T), Th)
    dxdt = -Ft / Fx
    ok2 = abs(xs[0] - 1) < 1e-20 and abs(Xh - 1) < 1e-15 and abs(dxdt + 1) < 1e-15 and abs(xmax - (mp.sqrt(17) - 1) / 2) < 1e-6
    report(3, ok1 and ok2, f"(1) f(a)={fa}, 최댓값 {mx}; (2) B(t=π/2) x={mp.nstr(Xh, 10)}, dx/dt={mp.nstr(dxdt, 12)}, x 최댓값≈{mp.nstr(xmax, 10)} ((√17-1)/2={mp.nstr((mp.sqrt(17)-1)/2, 10)})")


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
