#!/usr/bin/env python3
"""INHA-G26B (2026학년도 인하대 논술 자연계열 오후 일반) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "INHA-G26B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) f'=0 의 근 a_n, √n tan2a_n 극한 (1-2) (1+√n) b_n 극한 — n 기호로 독립 재계산 + 수치."""
    N = sp.symbols("N", positive=True)
    X = sp.symbols("X", positive=True)
    f = sp.sin(X) ** N * sp.cos(X)
    fp = sp.expand(sp.diff(f, X) * sp.sin(X) ** (1 - N), power_base=False)   # = N cos^2 - sin^2
    fp = sp.simplify(sp.powsimp(fp, force=True))
    c2 = sp.symbols("c2", positive=True)
    poly = sp.expand(fp.subs(sp.sin(X) ** 2, 1 - c2).subs(sp.cos(X) ** 2, c2))
    c2sol = sp.solve(sp.Eq(poly, 0), c2)[0]
    cos_a = sp.sqrt(c2sol)
    sin_a = sp.sqrt(1 - c2sol)
    tan_a = sp.simplify(sin_a / cos_a)
    L1 = sp.limit(sp.sqrt(N) * 2 * tan_a / (1 - tan_a**2), N, sp.oo)
    bn = sin_a**N * cos_a
    L2 = sp.limit((1 + sp.sqrt(N)) * bn, N, sp.oo)
    # 수치: n=10^6 에서 수치 최적화한 극값 위치로 확인 + 부호 변화(극대) 확인
    import scipy.optimize as so
    nn = 10**6
    res = so.minimize_scalar(lambda v: -(np.sin(v) ** nn * np.cos(v)), bounds=(1.0, np.pi / 2 - 1e-9), method="bounded", options={"xatol": 1e-14})
    num1 = np.sqrt(nn) * np.tan(2 * res.x)
    sgn = [float(fp.subs({N: 5, X: v})) for v in (0.3, 1.4)]
    ok = (sp.simplify(c2sol - 1 / (N + 1)) == 0 and L1 == -2 and sp.simplify(L2 - sp.exp(-sp.Rational(1, 2))) == 0
          and abs(num1 + 2) < 1e-2 and sgn[0] > 0 > sgn[1])
    report(1, ok, f"cos^2 a_n={c2sol}, (1-1) {L1} [수치 {num1:.5f}] (1-2) {L2}")


def p02():
    """(2-1) 좌표로 정사각형 조건 풀이 (2-2) 넓이 1 하 최대 (2-3) 닮은 작은 정사각형과의 차 최대."""
    a, b, c, r, s_ = sp.symbols("a b c r s", positive=True)
    s = sp.symbols("s", real=True)
    yP = a / b * s + a
    yQ = -a / c * r + a
    sol = sp.solve([sp.Eq(yP, yQ), sp.Eq(yQ, r - s)], [r, s], dict=True)[0]
    side = sp.simplify(yQ.subs(sol))
    area = sp.simplify(side**2)
    ok1 = sp.simplify(area - a**2 * (b + c) ** 2 / (a + b + c) ** 2) == 0
    # (2-2): b+c=x, a=2/x
    x_ = sp.symbols("x", positive=True)
    side2 = sp.simplify(side.subs(a, 2 / (b + c)).subs(c, x_ - b))
    crit = [v for v in sp.solve(sp.diff(side2, x_), x_) if v.is_positive]
    ok2 = len(crit) == 1 and sp.simplify(crit[0] - sp.sqrt(2)) == 0 and sp.diff(side2, x_, 2).subs(x_, crit[0]) < 0
    # (2-3): 삼각형 APQ 는 높이 a-side, 밑변 PQ=side → 같은 공식 적용(닮음 대신 직접)
    hA = 2 / x_
    PQ = side2
    # 직접: 삼각형 APQ 의 높이 = hA - side2, 밑변 = PQ, 내접 정사각형 한 변 = h*w/(h+w)
    h_ = sp.simplify(hA - side2)
    small = sp.simplify(h_ * PQ / (h_ + PQ))
    diff = sp.simplify(side2**2 - small**2)
    crit3 = [v for v in sp.solve(sp.diff(diff, x_), x_) if v.is_positive]
    xs = np.linspace(0.01, 10, 2000001)
    fd = sp.lambdify(x_, diff, "numpy")(xs)
    xbest = xs[np.argmax(fd)]
    d2 = [sp.nsimplify(sp.simplify(v**2)) for v in crit3]
    ok3 = (len(crit3) == 1 and sp.simplify(crit3[0] ** 2 - (-1 + sp.sqrt(17))) == 0
           and abs(xbest**2 - float(-1 + sp.sqrt(17))) < 1e-4
           and sp.simplify(small - 4 * x_ / (x_**2 + 2) ** 2) == 0)
    report(2, ok1 and ok2 and ok3, f"(2-1) {area} (2-2) BC={crit} (2-3) XY={sp.factor(small)}, d^2={d2} [수치 {xbest**2:.6f}]")


def p03():
    """(3-1)(3-2) 수치 적분과 대조 (3-3) f'(π/2), f(π) 수치 적분."""
    from scipy import integrate as si
    A = sp.symbols("A", positive=True)
    tt = sp.symbols("tt", real=True)
    F1 = sp.integrate(sp.cancel((1 - sp.sin(tt)) / sp.cos(tt) ** 2), (tt, 0, A))
    F1 = sp.simplify(F1)
    target1 = sp.tan(A) - sp.sec(A) + 1
    g = lambda u: u * np.sin(u) / (1 + np.sin(u))
    ok1 = ok2 = True
    for av in (0.3, 0.9, 1.4):
        n1 = si.quad(lambda u: 1 / (1 + np.sin(u)), 0, av)[0]
        ok1 &= abs(n1 - float(target1.subs(A, av))) < 1e-9 and abs(n1 - float(F1.subs(A, av))) < 1e-9
        n2 = si.quad(g, 0, av)[0] + si.quad(g, np.pi - av, np.pi)[0]
        ok2 &= abs(n2 - float((sp.pi * (A - sp.tan(A) + sp.sec(A) - 1)).subs(A, av))) < 1e-9
    fprime = (tt * sp.sin(tt) / (1 + sp.sin(tt))).subs(tt, sp.pi / 2)
    hh = sp.symbols("h", positive=True)
    fpi_sym = sp.limit((sp.pi * (A - sp.tan(A) + sp.sec(A) - 1)).subs(A, sp.pi / 2 - hh), hh, 0, "+")
    fpi_num = si.quad(g, 0, np.pi, limit=200)[0]
    ok3 = (sp.simplify(fprime - sp.pi / 4) == 0 and sp.simplify(fpi_sym - (sp.pi**2 / 2 - sp.pi)) == 0
           and abs(fpi_num - float(sp.pi**2 / 2 - sp.pi)) < 1e-9)
    report(3, ok1 and ok2 and ok3, f"(3-1) {F1} (3-3) f'(π/2)={fprime}, f(π)={fpi_sym} [수치 {fpi_num:.10f}]")


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
