#!/usr/bin/env python3
"""DGU-M14 (2014학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M14"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """s(x)=∫0^x 3^t(4^t−19·2^t+48)dt, [0,4] 최대점."""
    tt = sp.symbols("tt", real=True)
    integrand = 3**tt * (2**(2 * tt) - 19 * 2**tt + 48)
    sprime = integrand.subs(tt, x)
    crit = [c for c in sp.solve(sp.Eq(sprime, 0), x) if c.is_real and 0 <= float(c) <= 4]
    cands = [sp.Integer(0), sp.Integer(4)] + crit
    vals = {c: float(sp.Integral(integrand, (tt, 0, c)).evalf(30)) for c in cands}
    best = max(vals, key=vals.get)
    ok = sp.simplify(best - sp.log(3, 2)) == 0 or abs(float(best) - float(sp.log(3, 2))) < 1e-12
    report(1, ok, f"임계점 {crit}, 후보값 { {str(k): round(v, 4) for k, v in vals.items()} } → x=log₂3")


def p02():
    """행렬 A 로 정의된 C, D 의 경계 곡선과 둘러싼 넓이."""
    xx, yy, a = sp.symbols("xx yy a", real=True)
    A = sp.Matrix([[2, -1], [-4, 2]])
    eqC = A * sp.Matrix([xx**2, yy])
    yC = sp.solve(eqC[0], yy)[0]
    ok_C = sp.simplify(eqC[1].subs(yy, yC)) == 0
    imgD = A * sp.Matrix([a, a])
    yD = sp.simplify(imgD[1].subs(a, xx) if imgD[0] == a else imgD[1].subs(a, sp.solve(sp.Eq(imgD[0], xx), a)[0]))
    pts = sp.solve(sp.Eq(yC, yD), xx)
    lo, hi = min(pts), max(pts)
    area = sp.integrate(sp.Abs(yD - yC), (xx, lo, hi))
    ok = ok_C and sp.simplify(yC - 2 * xx**2) == 0 and sp.simplify(yD + 2 * xx) == 0 and area == sp.Rational(1, 3)
    report(2, ok, f"C: y={yC}, D: y={yD}, 교점 x={pts}, 넓이={area}")


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
