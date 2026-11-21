#!/usr/bin/env python3
"""DGU-G13 (2013학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G13"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """g'(t)=2t³−15t²+36t 의 [0,3] 최댓점."""
    xx = sp.symbols("xx", real=True)
    g = sp.integrate(2 * xx**3 - 15 * xx**2 + 36 * xx, (xx, 0, t))
    gp = sp.diff(g, t)
    crit = [c for c in sp.solve(sp.diff(gp, t), t) if 0 <= c <= 3]
    cands = sorted(set(crit + [sp.Integer(0), sp.Integer(3)]))
    best = max(cands, key=lambda c: gp.subs(t, c))
    report(1, best == 2, f"g'(t)={sp.expand(gp)}, 후보 { {int(c): int(gp.subs(t, c)) for c in cands} } → t={best}")


def p02():
    """y=x³/3−x²+9 (x≥0): y 최소 속도, 원점 접선(=y/x 최소) 경제속도."""
    xp = sp.symbols("xp", positive=True)
    y = xp**3 / 3 - xp**2 + 9
    c1 = [c for c in sp.solve(sp.diff(y, xp), xp)]
    cand1 = c1 + [sp.Integer(0)]
    ymin_at = min(cand1, key=lambda c: y.subs(xp, c))
    r = y / xp
    c2 = [c for c in sp.solve(sp.diff(r, xp), xp) if c.is_positive]
    # 원점 접선 조건과 일치 확인
    tang = [c for c in sp.solve(sp.Eq(sp.diff(y, xp) * (0 - xp) + y, 0), xp) if c.is_positive]
    grid = np.linspace(0.01, 20, 200000)
    rv = sp.lambdify(xp, r, "numpy")(grid)
    ok = ymin_at == 2 and c2 == [3] and tang == [3] and abs(grid[rv.argmin()] - 3) < 1e-3
    report(2, ok, f"연료소모 최소 속도 {ymin_at}, 경제속도(y/x 최소=원점 접선) {c2}")


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
