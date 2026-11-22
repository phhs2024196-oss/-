#!/usr/bin/env python3
"""CAU-M27C (2027학년도 중앙대 창의형 모의논술 자연계열(공통 문제 1~6)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M27C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """[1-1] 접선·교점 sympy, [1-2] 부호 변화 지점에서 나눠 A_n 정확 적분(일반 n) + 수치 절댓값 적분 대조 → 극한."""
    import mpmath as mp
    f = x ** 3 - x
    l = sp.diff(f, x).subs(x, 1) * (x - 1)
    roots = [r for r in sp.solve(sp.Eq(f, l), x) if r != 1]
    pts = [(r, f.subs(x, r)) for r in roots]
    ok1 = pts == [(-2, -6)]
    N = sp.symbols("N", positive=True)
    h = sp.expand(f - l)
    An = sp.integrate(-h, (x, -2 - 1 / N, -2)) + sp.integrate(h, (x, -2, -2 + 1 / N))
    # 부호 확인 및 수치 대조(n=7)
    nn = 7
    num = mp.quad(lambda t: abs(t ** 3 - 3 * t + 2), [-2 - 1 / nn, -2, -2 + 1 / nn])
    ok_num = abs(num - float(An.subs(N, nn))) < 1e-12
    lim = sp.limit(N ** 2 * An, N, sp.oo)
    report(1, ok1 and ok_num and lim == 9 and sp.simplify(An - (1 / (2 * N ** 4) + 9 / N ** 2)) == 0,
           f"[1-1] 교점 {pts} / [1-2] A_n={sp.simplify(An)}, lim n^2A_n={lim}")


def p02():
    """음함수 x^4+7x^2y^2+y^4=45 에서 dy/dx=-1 인 제1사분면 점을 연립으로 직접 풀기."""
    X, Y = sp.symbols("X Y", positive=True)
    F = X ** 4 + 7 * X ** 2 * Y ** 2 + Y ** 4 - 45
    dydx = -sp.diff(F, X) / sp.diff(F, Y)
    sol = sp.solve([F, sp.numer(sp.together(dydx + 1))], [X, Y], dict=True)
    xs = sorted({sp.nsimplify(s[X]) for s in sol}, key=float)
    ok = len(xs) == 3 and all(sp.simplify(a - b) == 0 for a, b in zip(xs, [1, 5 ** sp.Rational(1, 4), 2]))
    report(2, ok, f"a = {xs}")


def p03():
    """[3-1] g'(x) 부호 격자 확인, [3-2] ∫_0^{g(1)} g^{-1} = 1·g(1) − ∫_0^1 g (역함수 넓이 관계)로 a 를 수치 풀이."""
    import mpmath as mp
    import numpy as np
    mp.mp.dps = 30
    a, t = sp.symbols("a t", positive=True)
    g = x + a * sp.Integral(sp.exp(-t ** 2), (t, 0, x))
    gp = sp.diff(g, x).doit()
    xs = np.linspace(-50, 50, 100001)
    ok1 = all(float(gp.subs(a, av).subs(x, 0)) > 0 for av in (0.01, 1, 100)) and \
        all(np.min(1 + av * np.exp(-xs ** 2)) > 0 for av in (0.01, 1, 100))
    def G(av, xv):
        return xv + av * mp.quad(lambda s: mp.e ** (-s ** 2), [0, xv])
    def lhs(av):  # 역함수 정적분을 직접: ∫_0^{g(1)} g^{-1}(y) dy 를 수치 역함수로
        g1 = G(av, 1)
        inv = lambda yv: mp.findroot(lambda u: G(av, u) - yv, yv / (1 + av))
        return mp.quad(inv, [0, g1])
    aroot = mp.findroot(lambda av: lhs(av) - 1, 1.5)
    tgt = mp.e / (mp.e - 1)
    report(3, ok1 and abs(aroot - tgt) < 1e-12, f"[3-1] g'>0 (a=0.01,1,100 격자) / [3-2] a={mp.nstr(aroot, 15)} vs e/(e−1)")


def p04():
    """점 P_n, Q_n 을 정의대로 sympy 로 차례로 구해 삼각형 넓이 합(신발끈)."""
    an = sp.Integer(1)
    total = 0
    for _ in range(10):
        P = (an ** 2, an)
        X0 = sp.symbols("X0")
        b = sp.solve(sp.Eq(0 - P[1], -1 * (X0 - P[0])), X0)[0]
        Yv = sp.symbols("Yv", positive=True)
        an1 = [r for r in sp.solve(sp.Eq(Yv, Yv ** 2 - b), Yv) if r > 0][0]   # y = x - b, x = y^2
        P2 = (an1 ** 2, an1)
        S = sp.Abs((b - P[0]) * (P2[1] - P[1]) - (P2[0] - P[0]) * (0 - P[1])) / 2
        total += S
        an = an1
    report(4, total == 440, f"Σ S_n = {total}")


def p05():
    """[5-1] 두 조건 연립으로 a,b, [5-2] 구간 [0,π] 격자+임계점으로 최댓값."""
    import numpy as np
    a, b = sp.symbols("a b", real=True)
    f = sp.exp(x + sp.pi / 3) * (a * sp.sin(x) + b * sp.cos(x))
    sol = sp.solve([sp.Eq(f.subs(x, sp.pi / 6), sp.exp(sp.pi / 2)), sp.Eq(sp.diff(f, x).subs(x, sp.pi / 6), sp.exp(sp.pi / 2))], [a, b], dict=True)
    ok1 = len(sol) == 1 and sp.simplify(sol[0][a] - sp.Rational(1, 2)) == 0 and sp.simplify(sol[0][b] - sp.sqrt(3) / 2) == 0
    fs = f.subs(sol[0])
    fn = sp.lambdify(x, fs, "numpy")
    X = np.linspace(0, np.pi, 2000001)
    mx = fn(X).max()
    tgt = np.exp(3 * np.pi / 4) / np.sqrt(2)
    crit = sp.nsolve(sp.diff(fs, x), x, 1.3)
    ok2 = abs(mx - tgt) / tgt < 1e-10 and abs(float(crit) - 5 * np.pi / 12) < 1e-12
    report(5, ok1 and ok2, f"[5-1] {sol} / [5-2] 격자 최댓값 {mx:.12f} vs e^(3π/4)/√2 {tgt:.12f}, 임계점 {float(crit):.12f}=5π/12")


def p06():
    """[6-1] t=2,2.5,3 에서 세 직선이 각각 곡선과 한 점만 만남을 격자로 확인 + 접점 sympy, [6-2] S(t) 신발끈·정적분."""
    import numpy as np
    t = sp.symbols("t", positive=True)
    x0 = sp.symbols("x0")
    tp = sp.solve(sp.Eq(sp.exp(x0) * (t - x0) + sp.exp(x0), 0), x0)   # 접선 y=e^{x0}(x-x0)+e^{x0} 가 (t,0) 지남
    ok_tp = tp == [t + 1]
    xs = np.linspace(-30, 30, 600001)
    curve = np.exp(np.abs(xs))
    ok_one = True
    for tv in (2.0, 2.5, 3.0):
        lineA = (0 - 1) / (tv - 0) * (xs - tv)            # A(0,1) 지남
        lineC = np.exp(tv + 1) * (xs - tv)                # 접선
        for ln, x_touch in ((lineA, 0.0), (lineC, tv + 1)):
            d = curve - ln
            near = np.abs(xs - x_touch) < 1e-3
            ok_one &= bool(np.all(d[~near] > 0))          # 접점 밖에서는 만나지 않음
    A = (0, 1); B = (t, sp.exp(t)); C = (t + 1, sp.exp(t + 1))
    S = sp.simplify(((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / 2)
    ok_S = sp.simplify(S - ((sp.E - 1) * t * sp.exp(t) - sp.exp(t) + 1) / 2) == 0
    ok_pos = all(float(S.subs(t, tv)) > 0 for tv in (2, 2.5, 3))
    I = sp.simplify(sp.integrate(S, (t, 2, 3)))
    ok_I = sp.simplify(I - (2 * sp.E ** 4 - 4 * sp.E ** 3 + 2 * sp.E ** 2 + 1) / 2) == 0
    report(6, ok_tp and ok_one and ok_S and ok_pos and ok_I, f"[6-1] 접점 x={tp}, 한 점 조건 격자 확인 / [6-2] S(t)={S}, ∫={sp.expand(I)}")


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
