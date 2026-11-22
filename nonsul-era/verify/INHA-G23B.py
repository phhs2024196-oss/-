#!/usr/bin/env python3
"""INHA-G23B (2023학년도 인하대 자연계열 논술 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "INHA-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """(1-1) A=(t+1)e^{-t}/2 (1-2) 최댓값 9/(2e^2) (1-3) lim r/S = 1."""
    T = sp.symbols("T", positive=True)
    f = sp.exp(-x)
    P = (T, f.subs(x, T))
    m = sp.diff(f, x).subs(x, T)
    Qx = sp.solve(sp.Eq(P[1] + m * (x - T), 0), x)[0]
    Rx = sp.solve(sp.Eq(P[1] + (-1 / m) * (x - T), 0), x)[0]
    A = sp.simplify(Qx * P[1] / 2)
    ok1 = sp.simplify(A - (T + 1) * sp.exp(-T) / 2) == 0
    g = (2 * T - 1) * A
    crit = [c for c in sp.solve(sp.diff(g, T), T) if c >= 1]
    gmax = g.subs(T, crit[0])
    tg = np.linspace(1, 60, 200001)
    gv = sp.lambdify(T, g, "numpy")(tg)
    ok2 = sp.simplify(gmax - sp.Rational(9, 2) * sp.exp(-2)) == 0 and abs(gv.max() - float(gmax)) < 1e-9
    OP = sp.sqrt(T**2 + P[1] ** 2)
    PQ = sp.sqrt((Qx - T) ** 2 + P[1] ** 2)
    OQ = Qx
    r = 2 * A / (OP + PQ + OQ)
    S = sp.simplify((Qx - Rx) * P[1] / 2)
    L = sp.limit(r / S, T, sp.oo)
    ok3 = L == 1 and sp.simplify(S - (1 + sp.exp(-2 * T)) * sp.exp(-T) / 2) == 0
    report(1, ok1 and ok2 and ok3, f"(1-1) A={A} (1-2) t={crit[0]}, max={gmax} (1-3) R=({sp.simplify(Rx)},0), S={S}, lim={L}")


def p02():
    """(2-1) q<-128 또는 q>-20 (2-2)(a) 1+√(1-2p), 1-2√(1-2p) (b) lim q/A = -1/9."""
    q, p = sp.symbols("q p", real=True)
    h = x**3 - 3 * (x + 4) ** 2
    # 교점 개수 = x^3-3(x+4)^2-q 의 서로 다른 실근 개수 → q 격자에서 직접 셈
    def nroots(qq):
        return len(set(sp.Poly(h - qq, x).real_roots()))
    qs = [sp.Rational(k, 2) for k in range(-300, 41)]
    one = [qq for qq in qs if nroots(qq) == 1]
    ok1 = one == [qq for qq in qs if qq < -128 or qq > -20]
    # (a) 공통접선: x^3 = 3(x-p)^2+q, 3x^2 = 6(x-p) 연립, x>1
    sol = sp.solve([sp.Eq(x**3, 3 * (x - p) ** 2 + q), sp.Eq(3 * x**2, 6 * (x - p))], [x, q], dict=True)
    u = sp.sqrt(1 - 2 * p)
    pv = -5
    good = [s for s in sol if float(s[x].subs(p, pv)) > 1]
    t0, q0 = good[0][x], good[0][q]
    cubic = sp.expand(x**3 - 3 * (x - p) ** 2 - q0)
    PV = (-7, -2, sp.Rational(-1, 3), sp.Rational(1, 4))
    ok2 = all(abs(float((t0 - (1 + u)).subs(p, pv))) < 1e-12 for pv in PV)
    for pv in (-7, -2, sp.Rational(-1, 3), sp.Rational(1, 4)):
        coeffs = [complex(sp.N(sp.expand(cubic).subs(p, pv).coeff(x, k), 30)) for k in (3, 2, 1, 0)]
        rr = np.roots(coeffs)
        expv = [float((1 + u).subs(p, pv)), float((1 - 2 * u).subs(p, pv))]
        # 근 집합 = {중근 1+√(1-2p), 단근 1-2√(1-2p)}
        ok2 = ok2 and all(abs(r.imag) < 1e-5 for r in rr) and all(min(abs(r.real - e) for e in expv) < 1e-5 for r in rr) \
            and all(min(abs(r.real - e) for r in rr) < 1e-5 for e in expv)
    al, be = 1 + u, 1 - 2 * u
    A = sp.integrate((x - al) ** 2 * (x - be), (x, be, al))
    ok3a = all(abs(abs(float(A.subs(p, pv))) - 27 / 4 * (1 - 2 * pv) ** 2) < 1e-8 for pv in (-3, -0.5, 0.2))
    L = sp.limit(q0 / (sp.Rational(27, 4) * (1 - 2 * p) ** 2), p, -sp.oo)
    ok3 = ok3a and sp.simplify(L + sp.Rational(1, 9)) == 0 and all(abs(float((q0 - ((1 + u) ** 2 * (1 - 2 * u) - 3 * p**2)).subs(p, pv))) < 1e-9 for pv in PV)
    report(2, ok1 and ok2 and ok3, f"(2-1) q 격자 판정 일치={ok1} (2-2)(a) t={t0} ok={ok2} (b) q={sp.simplify(q0)}, A=27(1-2p)^2/4 수치 일치={ok3a}, lim={L}")


def p03():
    """(3-1) π/2<t<π (3-2) S=-√2π²/32+(√2/4+1)π-√2 (3-3) 최댓값 π."""
    def fpoly(tt, xx):
        return np.where(xx <= tt, np.sin(tt) * xx, tt * np.sin(tt) / (tt - np.pi) * (xx - np.pi))
    def count(tt):
        xs = np.linspace(0, np.pi, 400001)
        hv = fpoly(tt, xs) - xs * np.sin(xs)
        zs = {0.0, round(tt, 9), round(np.pi, 9)}
        eps = 1e-4
        for lo, hi in ((eps, tt - eps), (tt + eps, np.pi - eps)):
            if hi <= lo:
                continue
            msk = (xs > lo) & (xs < hi)
            s = np.sign(hv[msk])
            zs |= {round(float(v), 4) for v in xs[msk][1:][s[1:] * s[:-1] < 0]}
        return len(zs)
    ts = [(k + 0.37) * np.pi / 64 for k in range(0, 64)]          # 격자점과 겹치지 않게 비정렬
    got = [tt for tt in ts if count(tt) == 4]
    ok1 = got == [tt for tt in ts if tt > np.pi / 2]
    T = sp.pi / 4
    S2 = (sp.integrate(sp.sin(T) * x - x * sp.sin(x), (x, 0, T))
          + sp.integrate(x * sp.sin(x) - T * sp.sin(T) / (T - sp.pi) * (x - sp.pi), (x, T, sp.pi)))
    ok2 = sp.simplify(S2 - (-sp.sqrt(2) / 32 * sp.pi**2 + (sp.sqrt(2) / 4 + 1) * sp.pi - sp.sqrt(2))) == 0
    xs = np.linspace(0, np.pi, 20001)
    def S(tt):
        if tt == 0:
            return np.trapezoid(np.abs(xs * np.sin(xs)), xs)
        return np.trapezoid(np.abs(fpoly(tt, xs) - xs * np.sin(xs)), xs)
    tg = np.linspace(0, np.pi - 1e-3, 2001)
    sv = np.array([S(tt) for tt in tg])
    ok3 = tg[sv.argmax()] == 0 and abs(sv.max() - np.pi) < 1e-6
    report(3, ok1 and ok2 and ok3, f"(3-1) g(t)=4 ⇔ t>π/2 격자 일치={ok1} (3-2) S={sp.nsimplify(S2)}≈{float(S2):.6f} (3-3) max S≈{sv.max():.6f} at t={tg[sv.argmax()]}")


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
