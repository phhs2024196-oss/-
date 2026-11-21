#!/usr/bin/env python3
"""DGU-M22 (2022학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-M22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """포물선 y^2=4px: (1) HQ = FQ = |y1|/2 + 2p^2/|y1| (2) 최솟값 2|p|, P=(p, ±2p)."""
    ok = True
    for p in [sp.Integer(1), sp.Integer(-3), sp.Rational(5, 2)]:
        y1 = sp.symbols("y1", positive=True)
        for sgn in (1, -1):
            Y = sgn * y1
            X = Y**2 / (4 * p)
            # 접선 기울기: y^2=4px 음함수 미분 2y y'=4p → y'=2p/y
            xx, yy = sp.symbols("xx yy")
            slope = sp.solve(sp.diff(yy**2 - 4 * p * xx, xx) + sp.diff(yy**2, yy) * sp.Symbol("d"), sp.Symbol("d"))[0].subs(yy, Y)
            Qy = sp.simplify(Y + slope * (-p - X))
            H = (-p, Y); Q = (-p, Qy); Fp = (p, 0)
            HQ = sp.simplify(sp.Abs(Qy - Y))
            FQ = sp.simplify(sp.sqrt((Fp[0] - Q[0])**2 + (Fp[1] - Q[1])**2))
            target = y1 / 2 + 2 * p**2 / y1
            ok &= sp.simplify(HQ - target) == 0 and sp.simplify(FQ**2 - target**2) == 0
            crit = [c for c in sp.solve(sp.diff(target, y1), y1) if c > 0]
            mn = sp.simplify(target.subs(y1, crit[0]))
            Px = sp.simplify(X.subs(y1, crit[0])); Py = sp.simplify(Y.subs(y1, crit[0]))
            ok &= mn == 2 * abs(p) and Px == p and Py == sgn * 2 * abs(p)
    report(1, ok, "(1) HQ=FQ=|y1|/2+2p^2/|y1| (2) 최솟값 2|p|, P=(p,±2p) — p=1,-3,5/2 에서 확인")


def p02():
    """베이즈: C 키트 98/197, D 키트 99/297=1/3 → C 가 높다."""
    R = sp.Rational
    prior = R(1, 100)
    out = {}
    for kit, sens, fp in (("C", R(98, 100), R(1, 100)), ("D", R(99, 100), R(2, 100))):
        out[kit] = prior * sens / (prior * sens + (1 - prior) * fp)
    ok = out["C"] == R(98, 197) and out["D"] == R(1, 3) and out["C"] > out["D"]
    report(2, ok, f"C={out['C']}≈{float(out['C']):.4f}, D={out['D']}≈{float(out['D']):.4f}")


def p03():
    """(1) h'(t)=1/(3+cos h) 최소 → 최소 양수 a=6π (2) f(t)=∫0^{2π}|3x+sinx-t|dx 최소점 a=3π."""
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    h = sp.symbols("h", real=True)
    g = 3 * h + sp.sin(h)
    hp = 1 / sp.diff(g, h)                    # 역함수 미분
    # (1) h'(t) 최소 ⇔ cos h = 1 ⇔ h = 2kπ, t = g(h); 최소 양수 t
    cands = [g.subs(h, 2 * k * sp.pi) for k in range(1, 4)]
    a1 = min(c for c in cands if c > 0)
    hmin = min(float(hp.subs(h, v)) for v in np.linspace(0.01, 20, 4001))
    ok1 = a1 == 6 * sp.pi and abs(float(hp.subs(h, 2 * sp.pi)) - hmin) < 1e-6
    # 수치: t∈(0, a1) 에서는 h'(t) 최솟값 미달성
    gf = lambda hh: 3 * hh + np.sin(hh)
    ts = np.linspace(0.01, float(6 * sp.pi) - 0.01, 3000)
    hs = [brentq(lambda hh: gf(hh) - tt, 0, 10) for tt in ts]
    ok1 &= min(1 / (3 + np.cos(hh)) for hh in hs) > 0.25 + 1e-9
    # (2)
    F = lambda tt: quad(lambda xx: abs(3 * xx + np.sin(xx) - tt), 0, 2 * np.pi, limit=200, points=[brentq(lambda hh: gf(hh) - tt, 0, 2 * np.pi)] if 0 < tt < 6 * np.pi else None)[0]
    grid = np.linspace(0, 6 * np.pi, 6001)
    vals = [F(tt) for tt in grid]
    tmin = grid[int(np.argmin(vals))]
    ok2 = abs(tmin - 3 * np.pi) < 6 * np.pi / 6000 + 1e-9
    report(3, ok1 and ok2, f"(1) a={a1} (2) 수치 argmin={tmin:.6f}, 3π={3*np.pi:.6f}, f(3π)={F(3*np.pi):.6f}")


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
