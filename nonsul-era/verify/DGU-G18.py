#!/usr/bin/env python3
"""DGU-G18 (2018학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G18"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """T(x)=√(a²+x²)/v+(b−x), 0≤x≤b 의 최소점 x(v) — 경우 분류 공식과 무작위 수치 최소 비교."""
    a, b, v, X = sp.symbols("a b v X", positive=True)
    T = sp.sqrt(a**2 + X**2) / v + (b - X)
    crit = sp.solve(sp.Eq(sp.diff(T, X), 0), X)
    # sympy 임계점(v<1 일 때) = va/√(1−v²)
    x0 = sp.simplify(crit[0]) if crit else None
    ok_sym = x0 is not None and all(abs(float((x0 - v * a / sp.sqrt(1 - v**2)).subs({a: av, v: vv}))) < 1e-12
                                    for av, vv in [(1, 0.3), (2.5, 0.7), (0.4, 0.95)])

    def formula(av, bv, vv):
        return vv * av / np.sqrt(1 - vv**2) if vv <= bv / np.sqrt(av**2 + bv**2) else bv

    rng = np.random.default_rng(18)
    bad = 0
    for _ in range(300):
        av, bv = rng.uniform(0.1, 5, 2)
        vv = rng.uniform(0.05, 1.8)
        g = np.linspace(0, bv, 200001)
        Tg = np.sqrt(av**2 + g**2) / vv + (bv - g)
        if abs(g[Tg.argmin()] - formula(av, bv, vv)) > 1e-3 * max(1, bv):
            bad += 1
    report(1, ok_sym and bad == 0, f"임계점 {x0}; v≤b/√(a²+b²) → va/√(1−v²), 그 외 → b (무작위 300건 불일치 {bad})")


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
