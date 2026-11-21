#!/usr/bin/env python3
"""DGU-M17 (2017학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M17"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """연속 + 넓이 1 → a, c, E(X)."""
    a, c = sp.symbols("a c", positive=True)
    sol = sp.solve([sp.Eq(a * 10, c), sp.Eq(sp.integrate(a * x, (x, 0, 10)) + sp.integrate(c, (x, 10, 20)), 1)], [a, c], dict=True)[0]
    E = sp.integrate(x * sol[a] * x, (x, 0, 10)) + sp.integrate(x * sol[c], (x, 10, 20))
    ok = sol[a] == sp.Rational(1, 150) and sol[c] == sp.Rational(1, 15) and E == sp.Rational(110, 9)
    report(1, ok, f"a={sol[a]}, c={sol[c]}, E(X)={E}")


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
