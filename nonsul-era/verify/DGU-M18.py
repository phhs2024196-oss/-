#!/usr/bin/env python3
"""DGU-M18 (2018학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M18"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p03():
    """부피 V 일정, 겉넓이 2πr²+2πrh 최소 → r:h."""
    r, h, V = sp.symbols("r h V", positive=True)
    hs = V / (sp.pi * r**2)
    A = 2 * sp.pi * r**2 + 2 * sp.pi * r * hs
    rc = sp.solve(sp.diff(A, r), r)
    ratio = sp.simplify(hs.subs(r, rc[0]) / rc[0])
    second = sp.simplify(sp.diff(A, r, 2).subs(r, rc[0]))
    report(3, len(rc) == 1 and ratio == 2 and second.is_positive, f"r={rc[0]}, h/r={ratio}, A''>0 → r:h=1:2")


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
