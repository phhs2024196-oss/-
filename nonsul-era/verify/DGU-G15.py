#!/usr/bin/env python3
"""DGU-G15 (2015학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G15"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """[증명-수치확인] |가속도|≤2, v≥0, v(0)=0, x(8)=16 에서 x(9) 최대를 LP 로 계산 → 25 이하."""
    from scipy.optimize import linprog
    dt, N = 0.01, 900
    # 결정변수: 구간별 상수 가속도 a_0..a_{N-1}
    # v_k = dt*sum_{i<k} a_i, x_k = sum_{i<k}(v_i dt + a_i dt²/2) = sum_i a_i * w_k[i]
    def xcoef(k):
        w = np.zeros(N)
        for i in range(k):
            w[i] = dt * dt * (k - 1 - i) + dt * dt / 2
        return w
    A_ub, b_ub = [], []
    for k in range(1, N + 1):            # v_k ≥ 0  →  −dt*sum a_i ≤ 0
        row = np.zeros(N)
        row[:k] = -dt
        A_ub.append(row)
        b_ub.append(0.0)
    res = linprog(-xcoef(N), A_ub=np.array(A_ub), b_ub=b_ub, A_eq=[xcoef(800)], b_eq=[16.0],
                  bounds=[(-2, 2)] * N, method="highs")
    xmax = -res.fun
    # 등호 사례: [0,4] 정지, 이후 가속 2 → x(8)=16, x(9)=25
    tt = sp.symbols("tt", positive=True)
    pos = lambda T: sp.integrate(sp.Piecewise((0, tt < 4), (2 * (tt - 4), True)), (tt, 0, T))
    ok = res.status == 0 and xmax <= 25 + 1e-6 and abs(xmax - 25) < 1e-3 and pos(8) == 16 and pos(9) == 25
    report(1, ok, f"[증명-수치확인] x(8)=16 제약에서 max x(9)={xmax:.6f} ≤ 25, 등호 사례 x(9)={pos(9)} → 25 m 에 9초 이상 필요")


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
