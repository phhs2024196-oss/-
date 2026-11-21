#!/usr/bin/env python3
"""DGU-G16 (2016학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G16"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """y=x²/2−1 의 y축 회전 통, 부피 π tan²t → h(t), h'(t) 최댓값."""
    H, Y = sp.symbols("H Y", positive=True)
    tt = sp.symbols("tt", real=True)
    vol = sp.pi * sp.integrate(2 * Y + 2, (Y, 0, H))   # x² = 2y+2
    sols = [s for s in sp.solve(sp.Eq(vol, sp.pi * sp.tan(tt)**2), H)]
    hsol = [s for s in sp.solve(sp.Eq(H**2 + 2 * H, sp.tan(tt)**2), H)]
    hf = -1 + sp.sqrt(1 + sp.tan(tt)**2)
    ok_h = any(sp.simplify(s - hf) == 0 for s in hsol + sols)
    hp = sp.diff(hf, tt)
    grid = np.linspace(0, float(sp.pi / 4), 20001)
    hpv = sp.lambdify(tt, hp, "numpy")(grid)
    mono = np.all(np.diff(hpv) > -1e-12)
    mx = sp.nsimplify(sp.simplify(hp.subs(tt, sp.pi / 4)))
    hmax = float(hf.subs(tt, sp.pi / 4))
    ok = ok_h and mono and sp.simplify(mx - sp.sqrt(2)) == 0 and hmax <= 5
    report(1, ok, f"h(t)=sec t−1, h'(t) 증가, 최댓값 h'(π/4)={mx} (h(π/4)={hmax:.4f}≤5)")


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
