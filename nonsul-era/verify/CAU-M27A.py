#!/usr/bin/env python3
"""CAU-M27A (2027학년도 중앙대 일반형 모의논술 자연계열(공통 문제 1~4)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M27A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """카드 게임: 한 번의 시행 확률을 전수로 세고, 무승부 반복을 등비급수로 합산 → P(갑)=P(을)인 (n,k) (답 (11,6),(12,1))."""
    sols = []
    for n in range(11, 300):
        for k in range(1, 12):
            Ys = [y for y in range(1, 12) if y != k]
            win = lose = draw = 0
            for X in range(1, n + 1):
                for Y in Ys:
                    if X > Y: win += 1
                    elif X < Y: lose += 1
                    else: draw += 1
            tot = n * len(Ys)
            p, q, r = sp.Rational(win, tot), sp.Rational(lose, tot), sp.Rational(draw, tot)
            Pg, Pe = p / (1 - r), q / (1 - r)          # 등비급수 합
            if Pg == Pe:
                sols.append((n, k))
    report(1, sols == [(11, 6), (12, 1)], f"전수(n≤299) 해 {sols}")


def p02():
    """렘니스케이트 PA·PB=1: 원식에서 직접 y를 수치로 풀어 dy/dx=0 점과 ∫_a^1 x y^2 dx 확인."""
    import mpmath as mp
    mp.mp.dps = 30
    X, Y = sp.symbols("X Y", positive=True)
    F = ((X - 1) ** 2 + Y ** 2) * ((X + 1) ** 2 + Y ** 2) - 1
    # dy/dx = -F_x/F_y = 0 ⇔ F_x = 0 (F_y≠0) → 연립
    sol = sp.solve([F, sp.diff(F, X)], [X, Y], dict=True)
    pts = [(s[X], s[Y]) for s in sol]
    ok1 = len(pts) == 1 and sp.simplify(pts[0][0] - sp.sqrt(3) / 2) == 0 and sp.simplify(pts[0][1] - sp.Rational(1, 2)) == 0
    # 2-1 도함수 식 확인: 음함수 미분 결과와 예시답안 식 비교(수치 점)
    yfun = lambda x: mp.sqrt(mp.findroot(lambda s: ((x - 1) ** 2 + s) * ((x + 1) ** 2 + s) - 1, 0.3))
    x0 = mp.mpf("0.7"); y0 = yfun(x0)
    dydx_impl = -mp.diff(lambda x: ((x - 1) ** 2 + y0 ** 2) * ((x + 1) ** 2 + y0 ** 2) - 1, x0) / mp.diff(lambda y: ((x0 - 1) ** 2 + y ** 2) * ((x0 + 1) ** 2 + y ** 2) - 1, y0)
    dydx_off = (1 / y0) * (2 * x0 / mp.sqrt(4 * x0 ** 2 + 1) - x0)
    a = mp.sqrt(3) / 2
    I = mp.quad(lambda x: x * yfun(x) ** 2, [a, 1])
    target = 5 * mp.sqrt(5) / 12 - mp.mpf(173) / 192
    ok2 = abs(I - target) < mp.mpf("1e-20")
    report(2, ok1 and abs(dydx_impl - dydx_off) < 1e-15 and ok2,
           f"[2-1] (a,b)={pts}, y'식 일치 / [2-2] 수치적분 {mp.nstr(I, 15)} vs 5√5/12−173/192")


def p03():
    """[3-1] a의 최솟값을 이분법(구간 [1/4,16] 격자+끝점 최소)으로, [3-2] a를 x 격자 스캔(최댓값 도달 점 개수=3)으로 독립 탐색."""
    import numpy as np
    xs = np.concatenate([np.geomspace(0.25, 16, 200001), [0.25, 16.0]])
    def holds(a):
        k = 2.0 ** a
        lt = np.log(xs) / np.log(k)
        return np.min(np.abs((lt + 4) * (lt - 8))) >= 29
    lo, hi = 0.5, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if holds(mid) else (mid, hi)
    amin = (4 + 2 * np.sqrt(7)) / 3
    ok1 = abs(hi - amin) < 1e-7 and holds(amin + 1e-6) and not holds(amin - 1e-4)
    # 3-2
    X = np.linspace(0, 2 * np.pi, 400001)
    def nroots(a):
        f = np.abs(4 * np.sin(X) ** 2 - 4 * np.sin(X) - a)
        M = f.max(); on = f > M - 1e-6
        # 연결 덩어리 수(0과 2π는 같은 sin이지만 구간 [0,2π]에서는 다른 근)
        return int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
    hits = [a for a in np.arange(0.05, 20, 0.05) if nroots(a) == 3]
    xsym = sp.symbols("x")
    fx = lambda a, x: sp.Abs(4 * sp.sin(x) ** 2 - 4 * sp.sin(x) - a)
    A = sp.Rational(7, 2)
    vals = [sp.nsimplify(fx(A, r)) for r in (sp.pi / 6, 5 * sp.pi / 6, 3 * sp.pi / 2)]
    Mnum = max(float(fx(A, v)) for v in np.linspace(0, 2 * np.pi, 20001))
    ok2 = len(hits) == 1 and abs(hits[0] - 3.5) < 1e-9 and len(set(vals)) == 1 and abs(float(vals[0]) - Mnum) < 1e-9
    report(3, ok1 and ok2, f"[3-1] 이분법 a*={hi:.10f} vs (4+2√7)/3={amin:.10f} / [3-2] 세 근이 되는 a={hits}, f(π/6)=f(5π/6)=f(3π/2)={vals[0]}=M")


def p04():
    """[4-1] 접점을 접선 조건에서 수치로 풀어 m1m2 t^3 를 수치 적분, [4-2] a_n 을 격자 최소로 구해 합."""
    import mpmath as mp
    import numpy as np
    mp.mp.dps = 25
    def integrand(t):
        # 접점 x0: e^{x0^2} + 2x0 e^{x0^2}(t - x0) = 0 ⇔ 1 + 2x0(t-x0) = 0 (근을 수치로)
        g = lambda x0: 1 + 2 * x0 * (t - x0)
        r1, r2 = mp.polyroots([-2, 2 * t, 1])     # g(x0) = -2x0^2 + 2t x0 + 1
        assert abs(g(r1)) < 1e-20 and abs(g(r2)) < 1e-20
        m = [2 * r * mp.e ** (r ** 2) for r in (r1, r2)]
        assert abs(r1 - r2) > 1e-6
        return t ** 3 * m[0] * m[1]
    I = mp.quad(integrand, [0, 2])
    tgt = -(3 * mp.e ** 5 + mp.e)
    ok1 = abs(I - tgt) / abs(tgt) < 1e-15
    xs = np.concatenate([np.linspace(-2, 2, 400001), [-1.0, 0.0, 1.0]])
    s = 0.0
    for n in range(1, 28):
        e = n / 13
        s += np.min(np.abs(xs - 1) ** e + np.abs(xs + 1) ** e)
    c = 2 ** (1 / 13)
    tgt2 = (2 - c) / (c - 1) + 30
    ok2 = abs(s - tgt2) < 1e-9
    report(4, ok1 and ok2, f"[4-1] ∫={mp.nstr(I, 15)} vs −(3e^5+e) / [4-2] Σa_n 격자={s:.12f} vs 식 {tgt2:.12f}")


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
