#!/usr/bin/env python3
"""CAU-M22 (2022학년도 중앙대 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """주사위 (a,b) 4×8 전수, 원 x²+y²=a² 와 y=|b-4| 의 현 길이 l, √15≤l≤√20 (답 (2,4),(3,2),(3,6))."""
    sols = []
    for a in range(1, 5):
        for b in range(1, 9):
            k = abs(b - 4)
            if k < a:                      # 서로 다른 두 점
                l2 = 4 * (a * a - k * k)    # l^2
                if 15 <= l2 <= 20:
                    sols.append((a, b))
    # 오답 재현: b=4+k 만 택하면
    wrong = [(a, b) for (a, b) in sols if b >= 4]
    report(1, sols == [(2, 4), (3, 2), (3, 6)], f"해 {sols}, 한쪽만 택한 오답 {wrong}")


def p02():
    """(2-1) 정적분 1/3, (2-2) θ''(t0) = -2/(25e^4) — 좌표로 직접 θ(t) 정의해 수치·기호 확인."""
    X = sp.symbols("X")
    I1n = sp.N(sp.Integral((1 - sp.sin(X)) ** 2 / sp.cos(X) ** 3, (X, 0, sp.pi / 6)), 30)
    ok1 = abs(I1n - sp.Rational(1, 3)) < 1e-25
    tp = sp.symbols("tp", positive=True)
    u = sp.symbols("u", positive=True)
    s = sp.simplify(sp.integrate(sp.log(u + 1) / (u + 1), (u, 0, tp)))
    # θ = ∠APB, A(-2,2), B(-1,1), P(s,0) (s≥0): 외적·내적으로 tanθ, dot>0 이므로 θ=atan(cross/dot)
    PA = sp.Matrix([-2 - s, 2]); PB = sp.Matrix([-1 - s, 1])
    cross = PA[0] * PB[1] - PA[1] * PB[0]       # = s ≥ 0
    dot = PA.dot(PB)
    th = sp.atan(cross / dot)
    import mpmath as mp
    thf = sp.lambdify(tp, th, "mpmath")
    ts = [mp.mpf(i) / 100 for i in range(1, 3000)]
    tmax = max(ts, key=lambda z: thf(z))
    t0 = sp.exp(2) - 1
    ok_t0 = abs(float(tmax) - float(t0)) < 0.02
    d2n = sp.N(sp.diff(th, tp, 2).subs(tp, t0), 30)
    ok2 = abs(d2n - sp.N(-2 / (25 * sp.exp(4)), 30)) < 1e-25
    report(2, ok1 and ok_t0 and ok2, f"(1) {I1n:.12f} / (2) t_max≈{float(tmax):.3f} (e²-1={float(t0):.3f}), θ''(t0)={float(d2n):.10f}")


def p03():
    """(3-1) R 넓이: (a,b) 격자에서 f'≥0·g'≥0 판정 후 경계 적분, (3-2) A(t) 수치 적분 후 범위."""
    a, b = sp.symbols("a b", real=True)
    # f'≥0 ∀x ⇔ a²≤3b ; g'≥0 ∀x 를 t=x²≥0 최소값으로 직접 판정하는 함수
    def g_ok(av, bv):
        tv = max(0.0, 3 * bv / 10)
        return 5 * tv * tv - 3 * bv * tv + av >= -1e-12
    # 몬테카를로 대신 a 격자별 b 구간 길이를 이분법으로 구해 적분 근사
    import numpy as np
    N = 4000
    amax = 4.0
    tot = 0.0
    for i in range(N):
        av = (i + 0.5) * amax / N
        lo = av * av / 3
        # b 상한: g_ok 가 참인 최대 b (단조)
        hi_lo, hi_hi = lo, 10.0
        if not g_ok(av, lo):
            continue
        for _ in range(60):
            m = (hi_lo + hi_hi) / 2
            if g_ok(av, m):
                hi_lo = m
            else:
                hi_hi = m
        tot += (hi_lo - lo) * amax / N
    exact = sp.integrate(sp.sqrt(20 * a) / 3 - a ** 2 / 3, (a, 0, sp.cbrt(20)))
    ok1 = abs(tot - 20 / 9) < 1e-3 and sp.nsimplify(sp.simplify(exact)) == sp.Rational(20, 9)
    # (3-2)
    T = sp.symbols("T", positive=True)
    p = sp.symbols("p")
    roots = sp.solve(sp.Eq(-3 * T ** 2 + 4 * T - 1, 2 * p * T - p ** 2), p)
    r1, r2 = sorted(roots, key=lambda r: r.subs(T, 3))
    area = (sp.integrate(x ** 2, (x, r1, r2))
            - sp.integrate(2 * r1 * x - r1 ** 2, (x, r1, T))
            - sp.integrate(2 * r2 * x - r2 ** 2, (x, T, r2)))
    area = sp.factor(sp.expand(area))
    ok_area = sp.simplify(area - sp.Rational(2, 3) * (2 * T - 1) ** 3) == 0
    sol = sp.solve_univariate_inequality(sp.And(area > 18, area < 486).args[0], T, relational=False) \
        .intersect(sp.solve_univariate_inequality(area < 486, T, relational=False)) \
        .intersect(sp.Interval.open(sp.Rational(1, 2), sp.oo))
    ok2 = ok_area and sol == sp.Interval.open(2, 5)
    report(3, ok1 and ok2, f"(1) 수치넓이 {tot:.5f}, 기호 {exact} / (2) A(t)={area}, 범위 {sol}")


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
