#!/usr/bin/env python3
"""DGU-M16B (2016학년도 동국대 모의논술(2)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M16B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """현수선 길이 e^a−e^{-a}, 닮음비 k 조건 → k, 거리 2ka."""
    a, k = sp.symbols("a k", positive=True)
    f = (sp.exp(x) + sp.exp(-x)) / 2 - 1
    L = sp.integrate(sp.sqrt(sp.simplify(1 + sp.diff(f, x)**2)).rewrite(sp.exp), (x, -a, a))
    L = sp.simplify(L)
    ok_L = sp.simplify(L - (sp.exp(a) - sp.exp(-a))) == 0 or abs(float((L - (sp.exp(a) - sp.exp(-a))).subs(a, 0.7))) < 1e-12
    sol = sp.solve([sp.Eq(k * (sp.exp(a) - sp.exp(-a)), 6), sp.Eq(k * f.subs(x, a), 1)], [k, a], dict=True)
    sol = [s for s in sol if s[k].is_positive and s[a].is_positive]
    kk, aa = sol[0][k], sol[0][a]
    D = sp.simplify(2 * kk * aa)
    ok = ok_L and len(sol) == 1 and kk == 4 and sp.simplify(aa - sp.log(2)) == 0 and sp.simplify(D - 8 * sp.log(2)) == 0
    report(1, ok, f"길이={L}, k={kk}, a={aa}, 거리={D}≈{float(D):.4f}")


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
