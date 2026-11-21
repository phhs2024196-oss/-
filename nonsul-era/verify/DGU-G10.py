#!/usr/bin/env python3
"""DGU-G10 (2010학년도 동국대 논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-G10"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """[증명-수치확인] 일반항 a_n=2n−1 과 점화식 a_1=1, a_n=a_{n−1}+2 가 같은 수열, Σ=10000; 피보나치 점화식=비네 공식."""
    a = [None, 1]
    for k in range(2, 201):
        a.append(a[-1] + 2)
    ok1 = all(a[k] == 2 * k - 1 for k in range(1, 201))
    S = sp.summation(2 * n - 1, (n, 1, 100))
    b = [None, 1, 1]
    for k in range(3, 61):
        b.append(b[-1] + b[-2])
    phi, psi = (1 + sp.sqrt(5)) / 2, (1 - sp.sqrt(5)) / 2
    ok2 = all(sp.simplify(sp.expand((phi**k - psi**k) / sp.sqrt(5)) - b[k]) == 0 for k in range(1, 31))
    report(1, ok1 and ok2 and S == 10000, f"[증명-수치확인] a_n 두 표현 일치(n≤200), Σ={S}, 비네 공식=점화식(n≤30)")


def p02():
    """b_n/b_{n-1} 극한 x 는 x=1/x+1 의 양근 (1+√5)/2."""
    xp = sp.symbols("xp")
    roots = sp.solve(sp.Eq(xp, 1 / xp + 1), xp)
    pos = [r for r in roots if r.is_positive]
    b = [1, 1]
    for _ in range(80):
        b.append(b[-1] + b[-2])
    ratio = sp.Rational(b[-1], b[-2])
    ok = len(pos) == 1 and sp.simplify(pos[0] - (1 + sp.sqrt(5)) / 2) == 0 and abs(float(ratio) - float(pos[0])) < 1e-15
    report(2, ok, f"근 {roots}, 양근 {pos[0]}≈{float(pos[0]):.6f}, b_81/b_80≈{float(ratio):.12f}")


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
