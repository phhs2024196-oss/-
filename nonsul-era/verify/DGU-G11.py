#!/usr/bin/env python3
"""DGU-G11 (2011학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G11"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p03():
    """a_{n+1}=2/3a_n+1/3b_n, b_{n+1}=1/3a_n+2/3b_n, a_1=10000, b_1=70000 의 극한."""
    M = sp.Matrix([[sp.Rational(2, 3), sp.Rational(1, 3)], [sp.Rational(1, 3), sp.Rational(2, 3)]])
    v1 = sp.Matrix([10000, 70000])
    nn = sp.symbols("nn", integer=True, positive=True)
    P, D = M.diagonalize()
    vn = P * sp.diag(*[D[i, i]**(nn - 1) for i in range(2)]) * P.inv() * v1
    lim = [sp.limit(vn[i], nn, sp.oo) for i in range(2)]
    it = v1
    for _ in range(60):
        it = M * it
    ok = lim == [40000, 40000] and all(abs(float(it[i]) - 40000) < 1e-20 * 1e10 for i in range(2))
    report(3, ok, f"a_n={sp.simplify(vn[0])}, 극한 x={lim[0]}, y={lim[1]}")


def p04():
    """이동 규칙에서 P, Q 를 세우고 R=QP (c₁,d₁ 기호로 직접 이동 시뮬레이션과 대조)."""
    c1, d1 = sp.symbols("c1 d1")
    def move(c, d, pc, pd):           # C 의 pc 가 D 로, D 의 pd 가 C 로 동시에
        return sp.expand(c - pc * c + pd * d), sp.expand(d - pd * d + pc * c)
    c2, d2 = move(c1, d1, sp.Rational(1, 5), sp.Rational(2, 5))
    c3, d3 = move(c2, d2, sp.Rational(3, 5), sp.Rational(2, 5))
    P = sp.Matrix([[c2.coeff(c1), c2.coeff(d1)], [d2.coeff(c1), d2.coeff(d1)]])
    Qm = sp.Matrix([[sp.Rational(2, 5), sp.Rational(2, 5)], [sp.Rational(3, 5), sp.Rational(3, 5)]])
    c2s, d2s = sp.symbols("c2s d2s")
    c3q, d3q = move(c2s, d2s, sp.Rational(3, 5), sp.Rational(2, 5))
    Qcalc = sp.Matrix([[c3q.coeff(c2s), c3q.coeff(d2s)], [d3q.coeff(c2s), d3q.coeff(d2s)]])
    R = sp.Matrix([[c3.coeff(c1), c3.coeff(d1)], [d3.coeff(c1), d3.coeff(d1)]])
    ok = R == Qcalc * P and P == sp.Matrix([[sp.Rational(4, 5), sp.Rational(2, 5)], [sp.Rational(1, 5), sp.Rational(3, 5)]]) \
        and Qcalc == Qm and R == Qm and (P * Qcalc) != R
    report(4, ok, f"P={P.tolist()}, Q={Qcalc.tolist()}, R=QP={R.tolist()} (PQ={(P*Qcalc).tolist()} 와 다름)")


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
