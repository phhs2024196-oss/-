#!/usr/bin/env python3
"""CAU-M27D (2027학년도 중앙대 창의형 모의논술 의·약학부 추가 문제 7·8) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-M27D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p07():
    """판별식=0 으로 P(t,3)에서의 접선 기울기를 구하고, 두 접선의 사잇각이 45°가 되는 t^2 를 수치 탐색."""
    import mpmath as mp
    mp.mp.dps = 30
    m, T = sp.symbols("m T", real=True)
    X = sp.symbols("X")
    line = 3 + m * (X - T)
    quad = sp.expand((X ** 2 / 6 + line ** 2 / 3 - 1) * 6)
    disc = sp.discriminant(sp.Poly(quad, X).as_expr(), X)
    def acute(tv):
        ms = [complex(r) for r in sp.Poly(disc.subs(T, tv), m).nroots(n=30)]
        ms = [r.real for r in ms if abs(r.imag) < 1e-20]
        a1, a2 = (mp.atan(v) for v in ms)
        d = abs(a1 - a2) % mp.pi
        return min(d, mp.pi - d)
    t2 = mp.findroot(lambda s: acute(mp.sqrt(s)) - mp.pi / 4, 19)
    tgt = 6 + 6 * mp.sqrt(5)
    # 유일성: t^2 를 0.5~200 에서 훑어 45°를 지나는 곳 개수
    import numpy as np
    grid = np.linspace(0.5, 200, 4000)
    vals = [float(acute(mp.sqrt(g)) - mp.pi / 4) for g in grid if abs(g - 6) > 1e-3]
    crossings = sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
    report(7, abs(t2 - tgt) < 1e-12 and crossings == 1, f"t^2={mp.nstr(t2, 18)} vs 6+6√5, 45° 통과 {crossings}회")


def p08():
    """단위원 위 세 점 격자 전수로 AB·AC 최솟값과 그때의 넓이."""
    import numpy as np
    n = 720
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    A = np.array([1.0, 0.0])                       # 회전 대칭으로 A 고정
    Bx, By = np.cos(th)[:, None], np.sin(th)[:, None]
    Cx, Cy = np.cos(th)[None, :], np.sin(th)[None, :]
    dot = (Bx - A[0]) * (Cx - A[0]) + (By - A[1]) * (Cy - A[1])
    i, j = np.unravel_index(np.argmin(dot), dot.shape)
    mn = dot[i, j]
    area = abs((Bx[i, 0] - A[0]) * (Cy[0, j] - A[1]) - (Cx[0, j] - A[0]) * (By[i, 0] - A[1])) / 2
    # 정확값: 최소 배치(∠BOA=∠AOC=60°)에서 sympy
    a = sp.pi / 3
    Ae = sp.Matrix([1, 0]); Be = sp.Matrix([sp.cos(a), sp.sin(a)]); Ce = sp.Matrix([sp.cos(-a), sp.sin(-a)])
    de = sp.simplify((Be - Ae).dot(Ce - Ae))
    ae = sp.simplify(sp.Abs((Be - Ae)[0] * (Ce - Ae)[1] - (Ce - Ae)[0] * (Be - Ae)[1]) / 2)
    ok = abs(mn + 0.5) < 1e-9 and abs(area - np.sqrt(3) / 4) < 1e-9 and de == -sp.Rational(1, 2) and sp.simplify(ae - sp.sqrt(3) / 4) == 0
    report(8, ok, f"격자 최솟값 {mn:.10f}, 넓이 {area:.10f}; 정확값 {de}, {ae}")


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
