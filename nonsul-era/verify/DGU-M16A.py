#!/usr/bin/env python3
"""DGU-M16A (2016학년도 동국대 모의논술(1)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M16A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """x=e^{-t}cos t, y=e^{-t}sin t: 속도·가속도·t=2 속력·[0,π] 이동거리."""
    X = sp.exp(-t) * sp.cos(t)
    Y = sp.exp(-t) * sp.sin(t)
    vx, vy = sp.diff(X, t), sp.diff(Y, t)
    ax, ay = sp.diff(vx, t), sp.diff(vy, t)
    sp_ = sp.simplify(sp.sqrt(sp.expand(vx**2 + vy**2)))
    ok1 = sp.simplify(vx + sp.exp(-t) * (sp.cos(t) + sp.sin(t))) == 0 and sp.simplify(vy - sp.exp(-t) * (sp.cos(t) - sp.sin(t))) == 0
    ok2 = sp.simplify(ax - 2 * sp.exp(-t) * sp.sin(t)) == 0 and sp.simplify(ay + 2 * sp.exp(-t) * sp.cos(t)) == 0
    s2 = sp.simplify(sp_.subs(t, 2))
    dist = sp.simplify(sp.integrate(sp_, (t, 0, sp.pi)))
    ok = ok1 and ok2 and sp.simplify(s2 - sp.sqrt(2) * sp.exp(-2)) == 0 and sp.simplify(dist - sp.sqrt(2) * (1 - sp.exp(-sp.pi))) == 0
    report(1, ok, f"속력={sp_}, t=2 속력={s2}, 거리={dist}")


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
